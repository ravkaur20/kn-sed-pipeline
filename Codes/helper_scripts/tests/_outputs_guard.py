"""Keep the test suite from writing into the repo's real ``Outputs/`` tree (issue #9).

Two layers, installed once by ``_bootstrap_paths`` (which every test module imports):

1. **Write blocking.** ``open``/``os.open`` in a write mode, ``os.mkdir``/``makedirs``,
   ``remove``/``unlink``/``rmdir``/``rename``/``replace`` and ``shutil.rmtree``/``move``
   raise ``PermissionError`` when the target is under a protected root, so the
   offending test fails at the write.
2. **Snapshot backstop.** The protected trees are snapshotted (path, mtime, size) at
   install time and compared at interpreter exit. Any change (e.g. a write made by C
   code that bypasses the patched functions) is listed on stderr and the process
   exits with status 1.

Protected roots: ``<repo>/Outputs`` and, if the ``COCO_PATH`` env var is set,
``$COCO_PATH/Outputs``. Reads are never blocked.
"""

from __future__ import annotations

import atexit
import builtins
import io
import os
import shutil
import sys

_PROTECTED: list[str] = []
_SNAPSHOTS: dict[str, dict[str, tuple[int, int]]] = {}
_INSTALLED = False

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

_orig = {
    "open": builtins.open,
    "os.open": os.open,
    "os.mkdir": os.mkdir,
    "os.remove": os.remove,
    "os.unlink": os.unlink,
    "os.rmdir": os.rmdir,
    "os.rename": os.rename,
    "os.replace": os.replace,
    "shutil.rmtree": shutil.rmtree,
    "shutil.move": shutil.move,
}


def protected_roots(repo_root: str) -> list[str]:
    roots = [os.path.join(repo_root, "Outputs")]
    env = os.environ.get("COCO_PATH")
    if env:
        roots.append(os.path.join(env, "Outputs"))
    out: list[str] = []
    for r in roots:
        rp = os.path.realpath(r)
        if rp not in out:
            out.append(rp)
    return out


def is_protected(path) -> bool:
    """True if ``path`` (str/bytes/PathLike; fds are ignored) lies in a protected root."""
    if isinstance(path, int):
        return False
    try:
        p = os.fsdecode(os.fspath(path))
    except TypeError:
        return False
    rp = os.path.realpath(p)
    return any(rp == r or rp.startswith(r + os.sep) for r in _PROTECTED)


def _deny(what: str, path) -> None:
    raise PermissionError(
        "test tried to %s under the real Outputs/ tree: %s "
        "(tests must write to a temporary directory; see issue #9)" % (what, path)
    )


def _check(what: str, *paths, dir_fd=None) -> None:
    # Relative names with dir_fd can't be resolved here; the snapshot catches those.
    if dir_fd is not None:
        return
    for p in paths:
        if is_protected(p):
            _deny(what, p)


def _guarded_open(file, mode="r", *args, **kwargs):
    if any(c in mode for c in "wax+"):
        _check("open %r" % mode, file)
    return _orig["open"](file, mode, *args, **kwargs)


def _guarded_os_open(path, flags, *args, dir_fd=None, **kwargs):
    if flags & _WRITE_FLAGS:
        _check("os.open", path, dir_fd=dir_fd)
    if dir_fd is not None:
        kwargs["dir_fd"] = dir_fd
    return _orig["os.open"](path, flags, *args, **kwargs)


def _wrap_path_fn(name: str, n_paths: int = 1):
    fn = _orig[name]

    def guarded(*args, **kwargs):
        _check(name, *args[:n_paths], dir_fd=kwargs.get("dir_fd"))
        return fn(*args, **kwargs)

    guarded.__name__ = getattr(fn, "__name__", name)
    guarded.__doc__ = getattr(fn, "__doc__", None)
    return guarded


def snapshot(root: str) -> dict[str, tuple[int, int]]:
    """``{relpath: (mtime_ns, size)}`` for every file and dir under ``root``."""
    snap: dict[str, tuple[int, int]] = {}
    if not os.path.isdir(root):
        return snap
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                rel = os.path.relpath(e.path, root)
                if e.is_dir(follow_symlinks=False):
                    snap[rel + os.sep] = (0, 0)
                    stack.append(e.path)
                else:
                    snap[rel] = (st.st_mtime_ns, st.st_size)
    return snap


def diff_snapshots(before: dict, after: dict) -> list[str]:
    changes = []
    for k in sorted(set(before) | set(after)):
        if k not in after:
            changes.append("removed  " + k)
        elif k not in before:
            changes.append("added    " + k)
        elif before[k] != after[k]:
            changes.append("modified " + k)
    return changes


def _check_at_exit() -> None:
    lines = []
    for root, before in _SNAPSHOTS.items():
        for c in diff_snapshots(before, snapshot(root)):
            lines.append("  %s  (%s)" % (c, root))
    if not lines:
        return
    sys.stdout.flush()
    sys.stderr.write(
        "\n[outputs guard] FAIL: the test run changed the real Outputs/ tree "
        "(issue #9). %d change(s):\n" % len(lines)
    )
    sys.stderr.write("\n".join(lines[:50]) + "\n")
    if len(lines) > 50:
        sys.stderr.write("  ... and %d more\n" % (len(lines) - 50))
    sys.stderr.flush()
    os._exit(1)


def install(repo_root: str) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _PROTECTED[:] = protected_roots(repo_root)
    for r in _PROTECTED:
        _SNAPSHOTS[r] = snapshot(r)

    builtins.open = _guarded_open
    io.open = _guarded_open
    os.open = _guarded_os_open
    os.mkdir = _wrap_path_fn("os.mkdir")  # also covers os.makedirs
    os.remove = _wrap_path_fn("os.remove")
    os.unlink = _wrap_path_fn("os.unlink")
    os.rmdir = _wrap_path_fn("os.rmdir")
    os.rename = _wrap_path_fn("os.rename", 2)
    os.replace = _wrap_path_fn("os.replace", 2)
    shutil.rmtree = _wrap_path_fn("shutil.rmtree")
    shutil.move = _wrap_path_fn("shutil.move", 2)

    atexit.register(_check_at_exit)
