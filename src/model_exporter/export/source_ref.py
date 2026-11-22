# =============================================================================
# File: source_ref.py
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

"""Checks on the model reference a caller supplies.

A reference is a Hugging Face repository id (``org/name`` or ``name``), a local
directory, or, for the clone strategy only, an ``https://`` repository URL. It ends up
in file paths, log lines and a ``git`` command line, so it is checked once here.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from model_exporter.errors import InvalidModelReferenceError

# Hugging Face ids: one or two segments of letters, digits, ".", "_" and "-".
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?")
_HTTPS_URL = re.compile(r"https://[A-Za-z0-9.-]+(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]*)?")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def validate_model_reference(model_name: object) -> str:
    """Return *model_name* as a stripped string, or raise ``InvalidModelReferenceError``.

    Accepts repository ids and local paths. Rejects empty values, control characters
    and anything starting with ``-``, which a command line would read as an option.
    """
    text = str(model_name).strip() if model_name is not None else ""
    if not text:
        raise InvalidModelReferenceError("model_name cannot be empty")
    if _CONTROL.search(text):
        raise InvalidModelReferenceError("model_name contains control characters")
    if text.startswith("-"):
        raise InvalidModelReferenceError(f"model_name must not start with '-': {text!r}")
    return text


def is_repo_id(reference: str) -> bool:
    return bool(_REPO_ID.fullmatch(reference))


def hub_clone_url(reference: str) -> str:
    """The URL ``git clone`` is given for *reference*.

    A repository id becomes its ``https://huggingface.co/`` URL. An ``https://`` URL is
    used as it is. Everything else is refused: other schemes (``ext::``, ``file://``,
    ``ssh``), plain ``http``, option-like values and local paths (local directories are
    copied, not cloned).
    """
    text = validate_model_reference(reference)
    if _HTTPS_URL.fullmatch(text):
        return text
    if is_repo_id(text) and not os.path.exists(text):
        return f"https://huggingface.co/{text}"
    raise InvalidModelReferenceError(f"Not a Hugging Face repository id or https URL: {text!r}")


def copy_or_clone(source: str, dest: str) -> None:
    """Fill the directory *dest* with *source*: a copy of a local directory, or a
    shallow clone of a repository.

    ``--`` ends git's options and the URL comes from :func:`hub_clone_url`, so the
    reference can be neither an option nor a transport that runs commands.
    """
    if os.path.isdir(source):
        shutil.copytree(source, dest, dirs_exist_ok=True)
        return
    subprocess.check_call(["git", "clone", "--depth", "1", "--", hub_clone_url(source), dest])
