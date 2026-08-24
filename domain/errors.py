"""Expected framework failures, separated by responsibility."""


class SDEError(Exception):
    """Base class for errors callers may handle."""


class ConfigurationError(SDEError, ValueError):
    """Invalid configuration or component registration."""


class DataValidationError(SDEError, ValueError):
    """Input data violates a domain invariant."""


class CapabilityError(SDEError):
    """A component requires a capability the selected model does not expose."""


class NumericalError(SDEError):
    """A numerical procedure failed after its declared recovery policy."""


class ConvergenceError(SDEError):
    """An iterative algorithm cannot continue meaningfully."""
