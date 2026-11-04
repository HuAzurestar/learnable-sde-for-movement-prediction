"""Ten exploratory UO2 arms. Physical units are metres and seconds.

This versioned interface intentionally does not change the historical O1 models.
All stochastic steps are reparameterized; no trajectory is clipped or projected.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from infrastructure.research_store import digest

ARMS = ("M0-UO2", "M2-UO2", "A-S", "A-N", "B-S", "B-N",
        "C-T-S", "C-T-N", "C-R-S", "C-R-N")


class PilotError(ValueError):
    pass


@dataclass(frozen=True)
class PilotConfig:
    arm: str
    speed_limit: float  # Explicit frozen walking assumption; no default human limit.
    noise: float
    rho: float = 0.
    rate: float = 1.
    drive_limit: float = 1.
    omega: float = 0.
    seed: int = 2601

    def validate(self):
        if (self.arm not in ARMS or any(type(x) not in (int, float) or not math.isfinite(x)
                for x in (self.speed_limit, self.noise, self.rho, self.rate, self.drive_limit, self.omega))
                or min(self.speed_limit, self.rate, self.drive_limit) <= 0
                or min(self.noise, self.rho) < 0 or type(self.seed) is not int or self.seed != 2601):
            raise PilotError("INVALID_FROZEN_MODEL_CONFIG")


def saturate(x, limit):
    # Stable even for large finite affine inputs.
    scale = torch.maximum(x.abs().amax(dim=-1, keepdim=True), torch.ones_like(x[..., :1]))
    return limit * (x / scale) / torch.sqrt((1 / scale).square() + (x / scale).square().sum(-1, keepdim=True))


def turn(v):
    return torch.stack((-v[..., 1], v[..., 0]), dim=-1)


def rotate(v, angle):
    return torch.cos(angle)[..., None] * v + torch.sin(angle)[..., None] * turn(v)


def integration_grid(times,h):
    return sorted({round(i*h,10) for i in range(math.ceil(times[-1]/h))} |
                  {round(float(t),10) for t in times})


def radial_step(v, h, dB, U, rho):
    """Ito radial subflow: dv=-eta^2 v/2 dt+eta v dB.

    x=r^2/U^2: dx=2 rho x(1-x)dB. For y=logit(x),
    dy=2 rho^2 tanh(y/2)dt+2 rho dB. EM on y is consistent,
    smooth on the interior and maps into the ball for every finite increment.
    The zero and sphere radial subflows are absorbing; the zero derivative uses
    the limiting multiplicative radial flow. No C-T path envelope is claimed.
    """
    x = v.square().sum(-1) / U**2
    if bool((x > 1 + 1e-12).any()):
        raise PilotError("INITIAL_SUPPORT_REJECTED: radial state outside closed ball")
    delta=2*rho**2*(2*x-1)*h+2*rho*dB
    # Algebraically v*sqrt(exp(delta)/(1-x+x*exp(delta))). Both
    # absorbing boundaries retain the limiting Jacobian; neither branch
    # evaluates an overflowing positive exponential.
    positive=delta>=0
    dp=torch.where(positive,delta,torch.zeros_like(delta))
    dn=torch.where(positive,torch.zeros_like(delta),delta)
    plus=(x+(1-x)*torch.exp(-dp)).rsqrt()
    minus=torch.exp(dn/2)/torch.sqrt(1-x+x*torch.exp(dn))
    ratio=torch.where(positive,plus,minus)
    result=v*ratio[...,None]
    if not torch.isfinite(result).all():
        raise PilotError("NONFINITE_RADIAL_FLOW: extreme increment rejected")
    return result


class PilotModel(nn.Module):
    schema = "pirc26-walking-model-v1"

    def __init__(self, config, *, means=None, scales=None):
        super().__init__()
        config.validate()
        self.config = config
        self.family = ("M" if config.arm.startswith("M") else
                       "C-R" if config.arm.startswith("C-R") else
                       "C-T" if config.arm.startswith("C-T") else config.arm[0])
        self.neural = config.arm == "M2-UO2" or config.arm.endswith("-N")
        self.residual_enabled = True
        self.register_buffer("means", torch.tensor(means if means is not None else [0.] * 4, dtype=torch.float64))
        self.register_buffer("scales", torch.tensor(scales if scales is not None else [1.] * 4, dtype=torch.float64))
        if (self.means.shape != (4,) or self.scales.shape != (4,) or
                not torch.isfinite(self.means).all() or not torch.isfinite(self.scales).all() or
                (self.scales <= 0).any()):
            raise PilotError("INVALID_TRAIN_NORMALIZATION")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.affine = nn.Linear(4, 2, dtype=torch.float64)
            nn.init.zeros_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)
            if self.neural:
                self.layers = nn.ModuleList([nn.Linear(4, 16, dtype=torch.float64), nn.Linear(16, 2, dtype=torch.float64)])
                nn.init.xavier_uniform_(self.layers[0].weight)
                nn.init.zeros_(self.layers[0].bias)
                nn.init.zeros_(self.layers[1].weight)
                nn.init.zeros_(self.layers[1].bias)

    @property
    def noise_dim(self):
        return 1 if self.family == "C-T" else 2

    def physical_velocity(self, state):
        return saturate(state[..., 2:], self.config.speed_limit) if self.family == "B" else state[..., 2:]

    def initialize(self, physical_particles):
        x = torch.as_tensor(physical_particles, dtype=torch.float64).clone()
        if x.shape != (8, 4) or not torch.isfinite(x).all():
            raise PilotError("INVALID_EIGHT_CAUSAL_PARTICLES")
        speed2 = x[:, 2:].square().sum(-1)
        bound2 = self.config.speed_limit**2
        if self.family == "B":
            if bool((speed2 >= bound2).any()):
                raise PilotError("INITIAL_SUPPORT_REJECTED: B requires speed < V")
            x = torch.cat((x[:, :2], x[:, 2:] / torch.sqrt(bound2 - speed2)[:, None]), -1)
        elif self.family.startswith("C") and bool((speed2 > bound2).any()):
            raise PilotError("INITIAL_SUPPORT_REJECTED: C requires speed <= U")
        return x

    def raw_drive(self, state):
        physical = torch.cat((state[..., :2], self.physical_velocity(state)), -1)
        features = (physical - self.means) / self.scales
        value = self.affine(features)
        if self.neural and self.residual_enabled:
            r = features
            for i, layer in enumerate(self.layers):
                weight = layer.weight / torch.linalg.vector_norm(layer.weight).clamp(min=1.)
                r = F.linear(r, weight, layer.bias)
                r = torch.tanh(r)
            value = value + r
        return value

    def drive(self, state):
        raw = self.raw_drive(state)
        if self.family == "M":
            return raw
        return saturate(raw, self.config.drive_limit if self.family == "B" else self.config.speed_limit)

    def coefficients(self, state):
        """State-coordinate Ito drift and diffusion, including every correction."""
        c = self.config
        v = state[..., 2:]
        drive = self.drive(state)
        eye = torch.eye(2, dtype=state.dtype, device=state.device).expand(*v.shape[:-1], 2, 2)
        if self.family == "M":
            b, S = drive, c.noise * eye
        elif self.family == "A":
            b, S = c.rate * (drive - v) + c.omega * turn(v), c.noise * eye
        elif self.family == "B":
            b, S = -c.rate * v + drive, c.noise * eye
        else:
            eta = c.rho * (1 - v.square().sum(-1) / c.speed_limit**2) if self.family == "C-R" else torch.zeros_like(v[..., 0])
            b = c.rate * (drive - v) + c.omega * turn(v) - .5 * (c.noise**2 + eta.square())[..., None] * v
            S = c.noise * turn(v)[..., None]
            if self.family == "C-R":
                S = torch.cat((S, (eta[..., None] * v)[..., None]), -1)
        drift = torch.cat((self.physical_velocity(state), b), -1)
        diffusion = torch.cat((torch.zeros_like(S), S), -2)
        return drift, diffusion

    def step(self, state, h, dW):
        c = self.config
        old = self.physical_velocity(state)
        v, u = state[..., 2:], self.drive(state)
        e = math.exp(-c.rate * h)
        if self.family == "M":
            new = v + h * u + c.noise * dW
        elif self.family == "A":
            # omega is frozen zero in this pilot, so this is the exact frozen-u OU step.
            if c.omega != 0:
                raise PilotError("UNREGISTERED_A_TURNING: first pilot freezes omega=0")
            new = e * v + (1 - e) * u + c.noise * math.sqrt(-math.expm1(-2*c.rate*h)/(2*c.rate*h)) * dW
        elif self.family == "B":
            new = e * v + (1 - e) / c.rate * u + c.noise * math.sqrt(-math.expm1(-2*c.rate*h)/(2*c.rate*h)) * dW
        else:
            new = e * v + (1 - e) * u
            new = rotate(new, c.omega*h + c.noise*dW[..., 0])
            if self.family == "C-R":
                new = radial_step(new, h, dW[..., 1], c.speed_limit, c.rho)
        velocity = saturate(new, c.speed_limit) if self.family == "B" else new
        dp = h * old if self.family == "M" else .5*h*(old + velocity)
        result = torch.cat((state[..., :2] + dp, new), -1)
        if not torch.isfinite(result).all():
            raise PilotError("NONFINITE_PATH: no survivor-only scoring")
        if self.family in {"B", "C-T", "C-R"}:
            if bool((velocity.square().sum(-1) > c.speed_limit**2 * (1 + 1e-10)).any()):
                raise PilotError("NUMERICAL_BOUND_VIOLATION")
        return result

    def predict(self, particles, times, *, paths=64, seed=2603, h=.1, increments=None, cancelled=None):
        if (paths % 8 or not 8 <= paths <= 64 or not times or times[0] != 0 or
                any(not math.isfinite(t) for t in times) or any(b <= a for a,b in zip(times,times[1:]))
                or times[-1] > 60 or h not in (.1, .05)):
            raise PilotError("INVALID_PREDICTION_PLAN")
        state = self.initialize(particles).repeat(paths//8, 1)
        grid=integration_grid(times,h); count=len(grid)-1
        if increments is None:
            rng = torch.Generator().manual_seed(seed)
            intervals=torch.tensor([b-a for a,b in zip(grid,grid[1:])],dtype=torch.float64)
            increments = torch.randn(count, paths, self.noise_dim, generator=rng, dtype=torch.float64) * intervals.sqrt()[:,None,None]
        if increments.shape != (count, paths, self.noise_dim) or not torch.isfinite(increments).all():
            raise PilotError("INVALID_COUPLED_BROWNIAN")
        requested = {round(float(t),10) for t in times}
        outputs = [state]
        for k in range(count):
            if cancelled is not None and cancelled():
                raise PilotError("BUDGET_STOP")
            state = self.step(state, grid[k+1]-grid[k], increments[k])
            if grid[k+1] in requested:
                outputs.append(state)
        return torch.stack(outputs, 1)

    def checkpoint(self):
        return {"schema_version": self.schema, "config": asdict(self.config),
                "state": {k:v.detach().tolist() for k,v in self.state_dict().items()},
                "residual_enabled": self.residual_enabled}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if checkpoint.get("schema_version") != cls.schema:
            raise PilotError("CHECKPOINT_SCHEMA_MISMATCH")
        model = cls(PilotConfig(**checkpoint["config"]), means=checkpoint["state"]["means"], scales=checkpoint["state"]["scales"])
        state = {k:torch.tensor(v, dtype=torch.float64) for k,v in checkpoint["state"].items()}
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise PilotError("NONFINITE_CHECKPOINT")
        model.load_state_dict(state, strict=True)
        model.residual_enabled = checkpoint["residual_enabled"]
        return model

    def model_card(self):
        return {"schema_version": self.schema, "arm": self.config.arm, "config": asdict(self.config),
                "parameters": sum(p.numel() for p in self.parameters()),
                "physical_state": ["local_p_x_m", "local_p_y_m", "v_x_m_s", "v_y_m_s"],
                "internal_velocity": "latent_z" if self.family == "B" else "physical_v",
                "objective": "position-energy-u-v1", "noise_dim": self.noise_dim,
                "guarantee": "moment_only" if self.family == "A" else "none" if self.family == "M" else "speed_ball",
                "C_R_path_envelope": False, "checkpoint_hash": digest(self.checkpoint()),
                "independent_qualification": "required"}
