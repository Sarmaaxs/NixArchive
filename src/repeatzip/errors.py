class RepeatzipError(Exception):
    """Base class for all repeatzip errors."""

class WavFormatError(RepeatzipError):
    """Input WAV is not supported (must be 16-bit PCM)."""

class RpzFormatError(RepeatzipError):
    """Container is corrupt or has an unsupported version."""

class InvalidBlockSpec(RepeatzipError):
    """User-supplied or detected block is out of range / malformed."""

class ReconstructError(RepeatzipError):
    """Decoded output failed validation."""
