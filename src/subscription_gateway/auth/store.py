import asyncio
import fcntl
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from subscription_gateway.errors import GatewayError


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ext_agent_host_id: str
    client_id: str | None = None
    subject: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    id_token: str | None = None
    scopes: list[str] = []
    expires_at: float = 0


class CredentialStore(Protocol):
    def read(self) -> Credentials | None: ...
    def write(self, credentials: Credentials) -> None: ...
    def locked(self) -> AbstractAsyncContextManager[None]: ...


class FileCredentialStore:
    """POSIX store. Call write only while holding locked()."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "credentials.json"

    def prepare(self) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or self.directory.stat().st_uid != os.getuid():
            raise GatewayError(503, "unsafe_store", "Credential directory must be privately owned.")
        os.chmod(self.directory, 0o700)

    def read(self) -> Credentials | None:
        self.prepare()
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        with os.fdopen(fd) as handle:
            stat = os.fstat(handle.fileno())
            if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
                raise GatewayError(503, "unsafe_store", "Credential file permissions must be 0600.")
            try:
                return Credentials.model_validate(json.load(handle))
            except (ValueError, TypeError) as exc:
                raise GatewayError(503, "invalid_store", "Credential file is invalid.") from exc

    def write(self, credentials: Credentials) -> None:
        self.prepare()
        fd, name = tempfile.mkstemp(dir=self.directory, prefix=".credentials-")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(credentials.model_dump(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
            directory_fd = os.open(self.directory, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @asynccontextmanager
    async def locked(self) -> AsyncIterator[None]:
        self.prepare()
        fd = os.open(
            self.directory / "credentials.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            if os.fstat(fd).st_uid != os.getuid():
                raise GatewayError(503, "unsafe_store", "Credential lock must be privately owned.")
            os.fchmod(fd, 0o600)
            deadline = time.monotonic() + 30
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise GatewayError(503, "auth_busy", "Credential store is busy.") from None
                    await asyncio.sleep(0.025)
            yield
        finally:
            os.close(fd)
