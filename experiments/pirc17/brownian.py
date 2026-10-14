"""Shared Brownian increments for numerical refinement and paired forecasts.

Only times, dimensions, an origin identifier and a seed enter this driver: no observed positions,
labels, terrain or fitted parameters. Coarse increments are exact sums of the
same atomic Gaussian increments used by a refined integration schedule.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .origins import frozen_array
from .rollout import _horizons

BROWNIAN_VERSION="pirc17-coupled-brownian-v2"


def integration_grid(horizons_seconds, max_step_seconds, history_step_seconds):
    horizons=_horizons(horizons_seconds)
    if any(not np.isfinite(v) or v<=0 for v in (max_step_seconds,history_step_seconds)):
        raise ValueError("positive finite integration and history steps required")
    if horizons[-1]/min(max_step_seconds,history_step_seconds)>1_000_000:
        raise ValueError("coupling grid exceeds the bounded qualification budget")
    times=[0.]
    elapsed=0.
    tick=1
    next_history=float(history_step_seconds)
    for horizon in horizons:
        while elapsed<horizon:
            end=min(elapsed+max_step_seconds,float(horizon),next_history)
            if end<=elapsed:
                raise ValueError("step below floating-point time resolution")
            times.append(end)
            elapsed=end
            if elapsed==next_history:
                tick+=1
                next_history=tick*float(history_step_seconds)
    return times


class BrownianPath:
    def __init__(self,horizons_seconds,steps,*,history_step_seconds,particles,seed,stream_id,noise_dimensions=2):
        if not isinstance(stream_id,str) or not stream_id.strip():
            raise ValueError("nonempty immutable origin stream_id required")
        if type(seed) is not int or seed<0:
            raise ValueError("nonnegative integer seed required")
        if not steps or any(type(v) is not int or v<1 for v in (particles,noise_dimensions)):
            raise ValueError("steps and positive integer path dimensions required")
        times=sorted({t for step in steps for t in integration_grid(horizons_seconds,step,history_step_seconds)})
        # Bound memory independently of the time-grid bound above (float64 bytes).
        if len(times)*particles*noise_dimensions*8>512*1024**2:
            raise ValueError("coupled Brownian path exceeds 512 MiB")
        self.times=frozen_array(times)
        self.index={t:i for i,t in enumerate(times)}
        self.particles,self.noise_dimensions=particles,noise_dimensions
        # A separately registered RNG stream avoids sharing draws with a sampled
        # point-only initial velocity prior.
        stream_digest=hashlib.sha256(stream_id.encode("utf-8")).digest()
        stream_words=[int.from_bytes(stream_digest[i:i+4],"little") for i in range(0,32,4)]
        rng=np.random.default_rng(np.random.SeedSequence([seed,0x50495243,*stream_words]))
        # Particle-major draws preserve existing particle paths when the maximum
        # budget grows on the SAME time grid, including separate invocations.
        increments=rng.standard_normal((particles,len(times)-1,noise_dimensions)).transpose(1,0,2)*np.sqrt(np.diff(self.times))[:,None,None]
        self.values=np.concatenate((np.zeros((1,particles,noise_dimensions)),np.cumsum(increments,axis=0)))
        self.values.setflags(write=False)
        self.identity={"version":BROWNIAN_VERSION,"seed":seed,"stream_tag":0x50495243,
            "stream_id":stream_id,"stream_id_sha256":stream_digest.hex(),
            "seed_derivation":"SeedSequence([seed,0x50495243,*uint32_le(SHA256(UTF8(stream_id)))])",
            "draw_order":"particle,atomic_interval,noise_dimension",
            "numpy_version":np.__version__,"max_particles":particles,"noise_dimensions":noise_dimensions,
            "atomic_intervals":len(times)-1,"time_grid_sha256":hashlib.sha256(self.times.astype('<f8').tobytes()).hexdigest(),
            "path_sha256":hashlib.sha256(self.values.astype('<f8').tobytes()).hexdigest()}

    def __call__(self,start,end,particles,noise_dimensions):
        if start not in self.index or end not in self.index or start>=end:
            raise ValueError("Brownian request is outside the registered integration grid")
        if type(particles) is not int or not 0<particles<=self.particles or noise_dimensions!=self.noise_dimensions:
            raise ValueError("Brownian request dimensions differ from its registered path")
        return self.values[self.index[end],:particles]-self.values[self.index[start],:particles]
