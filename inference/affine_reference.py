"""Bounded rational enclosures for the declared constant-affine endpoint law.

No fitted model, qualification decision, process, ledger or budget API. Every
operation rounds *outward* on a fixed dyadic grid; mathematical remainder bounds
are retained. A content hash alone is not a proof: verification recomputes.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
import json
import math

from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import FrozenDynamicsPackage, _encode, content_hash
from domain.propagation import PropagationRequest


BITS = 160
GRID = 1 << BITS
MAX_INTEGER_BITS = 4096
MAX_TEMP_BITS = 16384
MAX_OPERATIONS = 200_000
MAX_BYTES = 128 * 1024
TAYLOR_TERMS = 32
MAX_DOUBLINGS = 60
CDF_TERMS = 256
ARCTAN_TERMS = 40


def algorithm_manifest():
    return {"id": "dyadic-affine-enclosure-v1", "fraction_bits": BITS,
        "integer_bits": MAX_INTEGER_BITS, "temporary_bits": MAX_TEMP_BITS,
        "operations": MAX_OPERATIONS, "certificate_bytes": MAX_BYTES,
        "taylor_terms": TAYLOR_TERMS, "doublings": MAX_DOUBLINGS,
        "cdf_terms": CDF_TERMS, "arctan_terms": ARCTAN_TERMS}


def _checked(value):
    value = Fraction(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_TEMP_BITS:
        raise NumericalError("reference rational intermediate exceeds fixed bit quota")
    return value


@dataclass(frozen=True)
class Interval:
    lo: Fraction
    hi: Fraction


class Arithmetic:
    def __init__(self):
        self.operations = 0

    def box(self, lo, hi=None):
        self.operations += 1
        if self.operations > MAX_OPERATIONS:
            raise NumericalError("reference arithmetic exceeds fixed operation quota")
        lo, hi = _checked(lo), _checked(lo if hi is None else hi)
        if lo > hi:
            raise NumericalError("empty reference enclosure")
        lower = (lo.numerator*GRID)//lo.denominator
        upper = -((-hi.numerator*GRID)//hi.denominator)
        if max(lower.bit_length(), upper.bit_length()) > MAX_INTEGER_BITS:
            raise NumericalError("reference enclosure exceeds fixed integer quota")
        return Interval(Fraction(lower, GRID), Fraction(upper, GRID))

    def add(self, x, y):
        return self.box(x.lo+y.lo, x.hi+y.hi)

    def sub(self, x, y):
        return self.box(x.lo-y.hi, x.hi-y.lo)

    def mul(self, x, y):
        products = (x.lo*y.lo, x.lo*y.hi, x.hi*y.lo, x.hi*y.hi)
        return self.box(min(products), max(products))

    def div(self, x, y):
        if y.lo <= 0 <= y.hi:
            raise NumericalError("reference interval division contains zero")
        reciprocals = (1/y.lo, 1/y.hi)
        return self.mul(x, self.box(min(reciprocals), max(reciprocals)))

    def sqrt(self, x):
        if x.lo < 0:
            raise NumericalError("reference square root has negative lower bound")
        def floor_root(q):
            return math.isqrt((q.numerator*GRID*GRID)//q.denominator)
        lower, upper = floor_root(x.lo), floor_root(x.hi)
        if Fraction(upper*upper, GRID*GRID) != x.hi:
            upper += 1
        return self.box(Fraction(lower, GRID), Fraction(upper, GRID))


def _zeros(a, rows, columns):
    return [[a.box(0) for _ in range(columns)] for _ in range(rows)]


def _transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def _add(a, x, y):
    return [[a.add(u, v) for u, v in zip(left, right)] for left, right in zip(x, y)]


def _matmul(a, x, y):
    return [[_dot(a, row, column) for column in zip(*y)] for row in x]


def _dot(a, x, y):
    result = a.box(0)
    for u, v in zip(x, y):
        if u.lo == u.hi == 0 or v.lo == v.hi == 0:
            continue  # Exact algebraic zero, never a numerical threshold.
        result = a.add(result, a.mul(u, v))
    return result


def _norm(matrix):
    return max(sum(max(abs(value.lo), abs(value.hi)) for value in row) for row in matrix)


def _short_exponential(a, matrix):
    size, norm = len(matrix), _norm(matrix)
    if norm > Fraction(1, 4):
        raise NumericalError("reference exponential was not scaled into its fixed domain")
    total = [[a.box(int(i == j)) for j in range(size)] for i in range(size)]
    term = total
    for k in range(1, TAYLOR_TERMS+1):
        term = [[a.div(value, a.box(k)) for value in row] for row in _matmul(a, term, matrix)]
        total = _add(a, total, term)
    # ||sum_{k=N+1}^infinity M^k/k!|| <= first omitted term /
    # (1 - ||M||/(N+2)). Each entry is bounded by the same norm bound.
    tail = _checked(norm**(TAYLOR_TERMS+1)/math.factorial(TAYLOR_TERMS+1)
                    / (1-norm/Fraction(TAYLOR_TERMS+2)))
    if all(value.lo == value.hi == 0 for row in term for value in row):
        tail = Fraction(0)  # Proven M^N=0, hence every subsequent power is zero.
    remainder = a.box(-tail, tail)
    # No positive-length walk in the matrix's nonzero graph => all positive
    # powers have that entry zero. The exponential entry is exactly I[i,j].
    # Interval nonzeros may overstate the graph, never remove a possible edge.
    reachable = [[value.lo != 0 or value.hi != 0 for value in row] for row in matrix]
    for k in range(size):
        for i in range(size):
            for j in range(size):
                reachable[i][j] = reachable[i][j] or (reachable[i][k] and reachable[k][j])
    return [[a.add(value, remainder) if reachable[i][j] else a.box(int(i == j))
             for j, value in enumerate(row)] for i, row in enumerate(total)]


def _pi(a):
    def arctan(q):
        total = Fraction(0)
        for k in range(ARCTAN_TERMS):
            total += (-1)**k*q**(2*k+1)/Fraction(2*k+1)
        tail = q**(2*ARCTAN_TERMS+1)/Fraction(2*ARCTAN_TERMS+1)
        # Even number of terms ends negative; next remainder is positive.
        return a.box(total, total+tail)
    return a.sub(a.mul(a.box(16), arctan(Fraction(1, 5))),
                 a.mul(a.box(4), arctan(Fraction(1, 239))))


def _negative_exponential(a, value):
    if value > 0:
        raise NumericalError("reference tail exponential requires a nonpositive argument")
    doublings = 0
    while abs(value) > Fraction(1, 4):
        value /= 2
        doublings += 1
        if doublings > MAX_DOUBLINGS:
            raise NumericalError("reference tail exceeds fixed scaling quota")
    result = _short_exponential(a, [[a.box(value)]])[0][0]
    for _ in range(doublings):
        result = a.mul(result, result)
        # Intersection with the proven range of exp(x), x <= 0.
        result = a.box(max(0, result.lo), min(1, result.hi))
    return result


def _cdf(a, x, denominator):
    if x < 0:
        positive = _cdf(a, -x, denominator)
        return a.box(1-positive.hi, 1-positive.lo)
    if x > 8:
        # Mills: P(Z>x) <= exp(-x²/2)/(x sqrt(2pi)). Beyond32
        # use monotonicity, avoiding unbounded powers of enormous arguments.
        cutoff = min(x, Fraction(32))
        density = a.div(_negative_exponential(a, -cutoff*cutoff/2), denominator)
        tail = a.div(density, a.box(cutoff))
        return a.box(max(0, 1-tail.hi), 1)
    term, total = a.box(x), a.box(0)
    square = a.box(x*x/2)
    for k in range(CDF_TERMS):
        total = a.add(total, term) if k % 2 == 0 else a.sub(total, term)
        term = a.mul(term, square)
        term = a.div(a.mul(term, a.box(2*k+1)), a.box((k+1)*(2*k+3)))
    # |x|<=8 and N=256 put the remaining alternating tail after its
    # decreasing-term region. The first omitted magnitude bounds its sum.
    total = a.add(total, a.box(-max(abs(term.lo), abs(term.hi)), max(abs(term.lo), abs(term.hi))))
    result = a.add(a.box(Fraction(1, 2)), a.div(total, denominator))
    return a.box(max(0, result.lo), min(1, result.hi))


def _strict_covariance(covariance):
    values = [[Fraction(value) for value in row] for row in covariance]
    if values != _transpose(values):
        raise DataValidationError("reference initial covariance must be exactly symmetric")
    # A real symmetric matrix is PSD iff all principal minors are nonnegative.
    for size in range(1, 5):
        for indices in combinations(range(4), size):
            determinant = Fraction(0)
            for perm in permutations(range(size)):
                inversions = sum(perm[i] > perm[j] for i in range(size) for j in range(i+1, size))
                product = Fraction((-1)**inversions)
                for row, column in enumerate(perm):
                    product = _checked(product*values[indices[row]][indices[column]])
                determinant = _checked(determinant+product)
            if determinant < 0:
                raise DataValidationError("reference initial covariance requires PSD projection")
    return values


def _rational_manifest(q):
    return [str(q.numerator), str(q.denominator)]


def _interval_manifest(interval):
    return [_rational_manifest(interval.lo), _rational_manifest(interval.hi)]


def _parse_interval(value):
    try:
        if type(value) is not list or len(value) != 2:
            raise ValueError
        bounds = []
        for pair in value:
            if type(pair) is not list or len(pair) != 2 or any(type(item) is not str or len(item) > 1235 for item in pair):
                raise ValueError
            numerator, denominator = (int(item) for item in pair)
            if pair != [str(numerator), str(denominator)] or denominator <= 0:
                raise ValueError
            if max(numerator.bit_length(), denominator.bit_length()) > MAX_INTEGER_BITS:
                raise ValueError
            q = Fraction(numerator, denominator)
            if _rational_manifest(q) != pair or q.denominator > GRID or q.denominator & (q.denominator-1):
                raise ValueError
            bounds.append(q)
        if bounds[0] > bounds[1]:
            raise ValueError
        return Interval(*bounds)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError("invalid bounded canonical reference interval") from exc


@dataclass(frozen=True)
class AffineReferenceCertificate:
    _document: bytes

    def manifest(self):
        if type(self._document) is not bytes or len(self._document) > MAX_BYTES:
            raise DataValidationError("reference certificate byte quota exceeded")
        try:
            result = json.loads(self._document)
            if type(result) is not dict or _encode(result) != self._document:
                raise ValueError
            remaining = 2048
            def visit(value, depth):
                nonlocal remaining
                remaining -= 1
                if remaining < 0 or depth > 10:
                    raise ValueError
                if type(value) is dict:
                    if len(value) > remaining:
                        raise ValueError
                    for key, child in value.items():
                        if len(key) > 128:
                            raise ValueError
                        visit(child, depth+1)
                elif type(value) is list:
                    if len(value) > remaining:
                        raise ValueError
                    for child in value:
                        visit(child, depth+1)
                elif type(value) is str and len(value) > 1235:
                    raise ValueError
            visit(result, 0)
            return result
        except (ValueError, TypeError, RecursionError) as exc:
            raise DataValidationError("reference certificate must be canonical JSON") from exc

    @property
    def certificate_hash(self):
        return content_hash(self.manifest())

    def functional_bounds(self):
        """Accessor only; untrusted certificates must be recomputed first."""
        try:
            return _parse_interval(self.manifest()["functional_bounds"])
        except (KeyError, TypeError) as exc:
            raise DataValidationError("reference functional interval missing") from exc


def bound_affine_reference(package, request):
    """Enclose moments/functional of the exact *declared* affine Gaussian law.

    Python floats in inputs denote their exact binary rational values. No claim
    covers parameter/model uncertainty, nonlinear closure or first passage.
    """
    if type(package) is not FrozenDynamicsPackage or type(request) is not PropagationRequest:
        raise DataValidationError("reference requires frozen affine dynamics and endpoint request")
    package.validate()
    request.validate()
    if package.package_hash != request.model_package_hash:
        raise DataValidationError("reference request/model content differs")
    initial_covariance = _strict_covariance(request.initial_covariance)
    p = package.manifest()["parameters"]
    a = Arithmetic()
    A = [[a.box(value) for value in row] for row in p["A"]]
    b = [[a.box(value)] for value in p["b"]]
    L = [[a.box(value) for value in row] for row in p["L"]]
    generator = _zeros(a, 9, 9)
    diffusion = _matmul(a, L, _transpose(L))
    for i in range(4):
        for j in range(4):
            generator[i][j], generator[i][j+4] = A[i][j], diffusion[i][j]
            generator[i+4][j+4] = a.sub(a.box(0), A[j][i])
        generator[i][8] = b[i][0]
    interval = Fraction(request.horizons[0])
    doublings = 0
    while True:
        boxed_interval = a.box(interval)
        scaled = [[a.mul(value, boxed_interval) for value in row] for row in generator]
        if _norm(scaled) <= Fraction(1, 4):
            break
        interval /= 2
        doublings += 1
        if doublings > MAX_DOUBLINGS:
            raise NumericalError("reference affine horizon exceeds fixed scaling quota")
    exponential = _short_exponential(a, scaled)
    F = [row[:4] for row in exponential[:4]]
    c = [[row[8]] for row in exponential[:4]]
    Q = _matmul(a, [row[4:8] for row in exponential[:4]], _transpose(F))
    for _ in range(doublings):
        Q = _add(a, Q, _matmul(a, _matmul(a, F, Q), _transpose(F)))
        c = _add(a, c, _matmul(a, F, c))
        F = _matmul(a, F, F)
    mean = _add(a, _matmul(a, F, [[a.box(value)] for value in request.initial_mean]), c)
    covariance = _add(a, _matmul(a, _matmul(a, F,
        [[a.box(value) for value in row] for row in initial_covariance]), _transpose(F)), Q)
    deterministic = not any(value for row in p["L"] for value in row) and not any(value for row in initial_covariance for value in row)
    if deterministic:
        # Exactly zero initial/diffusion covariance, not a PSD projection.
        covariance = _zeros(a, 4, 4)
        Q = _zeros(a, 4, 4)
    status = "BOUNDED"
    if request.functional == "endpoint-x":
        functional = mean[0][0]
    else:
        # Positive exact scaling leaves the event unchanged; avoid wasting the
        # absolute dyadic grid on an arbitrarily large normal-vector scale.
        exact_normal = [Fraction(value) for value in request.normal]
        scale = max(abs(value) for value in exact_normal)
        normal = [a.box(value/scale) for value in exact_normal]
        distance = a.sub(_dot(a, normal, [row[0] for row in mean]), a.box(Fraction(request.threshold)/scale))
        variance = _dot(a, normal, [row[0] for row in _matmul(a, covariance, [[value] for value in normal])])
        if variance.hi < 0:
            raise NumericalError("reference enclosure contradicts nonnegative projected variance")
        if variance.lo > 0:
            standardized = a.div(distance, a.sqrt(variance))
            denominator = a.sqrt(a.mul(a.box(2), _pi(a)))
            low, high = _cdf(a, standardized.lo, denominator), _cdf(a, standardized.hi, denominator)
            functional = a.box(low.lo, high.hi)
        elif variance.lo == variance.hi == 0:
            if (distance.lo >= 0 if request.closed else distance.lo > 0):
                functional = a.box(1)
            elif (distance.hi < 0 if request.closed else distance.hi <= 0):
                functional = a.box(0)
            else:
                status, functional = "UNRESOLVED_DEGENERATE_BOUNDARY", a.box(0, 1)
        else:
            status, functional = "UNRESOLVED_VARIANCE", a.box(0, 1)
    # Exact implementation/source binding, not an off-platform attestation.
    from experiments.pirc25.affine import code_hash
    document = {"schema_version": "affine-reference-certificate-v1", "algorithm": algorithm_manifest(),
        "code_hash": code_hash(), "model_package_hash": package.package_hash, "request_hash": request.request_hash,
        "scope": "declared-affine-Gaussian-endpoint-law", "scientific_qualification": False,
        "status": status, "doublings": doublings, "operations": a.operations,
        "units": "m" if request.functional == "endpoint-x" else "1",
        "functional_bounds": _interval_manifest(functional),
        "mean_bounds": [_interval_manifest(row[0]) for row in mean],
        "covariance_bounds": [[_interval_manifest(value) for value in row] for row in covariance],
        "transition_bounds": {"F": [[_interval_manifest(value) for value in row] for row in F],
            "offset": [_interval_manifest(row[0]) for row in c],
            "covariance": [[_interval_manifest(value) for value in row] for row in Q]}}
    encoded = _encode(document)
    if len(encoded) > MAX_BYTES:
        raise NumericalError("reference certificate exceeds fixed byte quota")
    return AffineReferenceCertificate(encoded)


def verify_affine_reference(certificate, package, request):
    """Recompute canonical evidence; resealing an invented interval cannot pass."""
    if type(certificate) is not AffineReferenceCertificate:
        raise DataValidationError("explicit reference certificate required")
    certificate.manifest()
    expected = bound_affine_reference(package, request)
    if certificate._document != expected._document:
        raise DataValidationError("reference certificate differs from recomputed bound source/request")
    return expected


def reference_error_bound(certificate, package, request, approximation):
    """Outward absolute-error bound for a finite float64 functional value."""
    try:
        finite = type(approximation) in (int, float) and math.isfinite(approximation)
    except OverflowError:
        finite = False
    if not finite:
        raise DataValidationError("finite reference approximation required")
    bounds = verify_affine_reference(certificate, package, request).functional_bounds()
    value = Fraction(approximation)
    error = max(abs(value-bounds.lo), abs(value-bounds.hi))
    try:
        result = float(error)
    except OverflowError as exc:
        raise NumericalError("reference error bound exceeds finite float range") from exc
    if not math.isfinite(result):
        raise NumericalError("reference error bound exceeds finite float range")
    if Fraction(result) < error:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise NumericalError("outward reference error bound exceeds finite float range")
    return result
