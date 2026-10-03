class GatewayError(Exception):
    """Safe, caller-visible error; never include upstream bodies or credentials."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def envelope(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}
