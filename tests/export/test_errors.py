# =============================================================================
# File: tests/export/test_errors.py
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""Exporter exceptions survive pickle and copy (they cross process boundaries)."""

import copy
import pickle

import pytest

from model_exporter.errors import ExportFailedError, ExportInProgressError, ExportValidationError

CASES = [
    ExportInProgressError("/out/model", pid=42),
    ExportFailedError("org/model", detail="first line\nlast line"),
    ExportValidationError("/out/model", returncode=3),
]


@pytest.mark.parametrize("error", CASES, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("clone", [lambda e: pickle.loads(pickle.dumps(e)), copy.copy], ids=["pickle", "copy"])
def test_round_trip_keeps_fields_and_message(error, clone):
    restored = clone(error)
    assert type(restored) is type(error)
    assert str(restored) == str(error)
    assert vars(restored) == vars(error)
