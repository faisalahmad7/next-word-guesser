"""Silence TensorFlow's startup noise without hiding real errors."""
from __future__ import annotations

import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_stderr():
    """Temporarily send stderr (including C++ library output) to the null device.

    Only wraps the noisy import/load step; stderr is restored afterwards, so
    exceptions raised later (or inside, once the block exits) still print normally.
    """
    try:
        sys.stderr.flush()
        saved_fd = os.dup(2)
        null_fd = os.open(os.devnull, os.O_WRONLY)
    except (OSError, ValueError, AttributeError):
        yield  # can't redirect; just run normally
        return
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        try:
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(null_fd)
