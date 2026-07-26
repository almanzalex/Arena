"""Bounded reads and durable transactional publication helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from arena.core.errors import IntegrityError, SchemaError, StoreError

T = TypeVar("T")

DEFAULT_MAX_BYTES = 10 * 1024 * 1024


def read_text_bounded(
    path: Path | str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    encoding: str = "utf-8",
) -> str:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise SchemaError(f"cannot stat input {source}: {exc}") from exc
    if size > max_bytes:
        raise SchemaError(
            f"input exceeds {max_bytes} byte limit: {source} has {size} bytes"
        )
    try:
        return source.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise SchemaError(f"input is not valid {encoding}: {source}") from exc
    except OSError as exc:
        raise SchemaError(f"cannot read input {source}: {exc}") from exc


def fsync_directory(path: Path) -> None:
    """Durably persist a directory entry where the platform permits it."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: Path | str,
    data: bytes,
    *,
    mode: int | None = None,
) -> Path:
    """Write a complete file, fsync it, atomically replace, then fsync its parent."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, destination)
        fsync_directory(destination.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return destination


def atomic_create_bytes(
    path: Path | str,
    data: bytes,
    *,
    mode: int | None = None,
) -> bool:
    """Atomically create immutable bytes without ever replacing a winner.

    Returns ``True`` when this caller published the file and ``False`` when an
    identical file already existed. A conflicting pre-existing file fails loud.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        try:
            os.link(tmp, destination)
            created = True
        except FileExistsError:
            verify_regular_file(destination, root=destination.parent)
            try:
                existing = destination.read_bytes()
            except OSError as exc:
                raise IntegrityError(
                    f"cannot verify pre-existing immutable file: {destination}"
                ) from exc
            if existing != data:
                raise IntegrityError(
                    f"immutable publication conflict at {destination}"
                ) from None
            created = False
        tmp.unlink(missing_ok=True)
        fsync_directory(destination.parent)
        return created
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def publish_directory(
    destination: Path | str,
    build: Callable[[Path], T],
    *,
    verify: Callable[[Path], None] | None = None,
    replace: bool = False,
) -> T:
    """Build and verify in same-parent staging, then publish atomically."""

    final = Path(destination)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.is_symlink():
        raise StoreError(f"refusing symlink publication destination: {final}")
    backup = final.with_name(f".{final.name}.previous")
    if backup.is_symlink():
        raise StoreError(f"refusing symlink publication backup: {backup}")
    if replace and backup.exists() and not final.exists():
        # Recover the last known-good publication after a process death between
        # moving the old destination aside and installing the new staging tree.
        os.replace(backup, final)
        fsync_directory(final.parent)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".staging", dir=final.parent)
    )
    try:
        result = build(staging)
        if verify is not None:
            verify(staging)
        if final.exists() and not replace:
            raise StoreError(
                f"refusing to replace existing output: {final}",
                code="OUTPUT_EXISTS",
                repair="Choose a new --out path or remove the prior output after verifying it.",
            )
        if replace:
            if backup.exists():
                raise StoreError(f"stale publication backup blocks output: {backup}")
            if final.exists():
                os.replace(final, backup)
                try:
                    os.replace(staging, final)
                    fsync_directory(final.parent)
                except BaseException:
                    os.replace(backup, final)
                    raise
                if backup.is_dir():
                    import shutil

                    shutil.rmtree(backup)
                else:
                    backup.unlink(missing_ok=True)
            else:
                os.replace(staging, final)
                fsync_directory(final.parent)
        else:
            os.rename(staging, final)
            fsync_directory(final.parent)
        return result
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_regular_file(path: Path, *, root: Path) -> None:
    """Refuse links and non-files, and assert a resolved path remains rooted."""

    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"expected a regular file: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise IntegrityError(f"path escapes artifact root: {path}") from exc
