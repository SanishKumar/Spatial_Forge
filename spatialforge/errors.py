"""Domain errors surfaced by the ScanSession boundary."""


class SessionValidationError(ValueError):
    """Raised when a session cannot satisfy the v0.1 contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


class SessionReplayError(RuntimeError):
    """Raised when validated replay input changes or becomes unreadable."""
