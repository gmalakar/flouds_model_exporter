# =============================================================================
# File: test_numeric.py
# Date: 2026-06-08
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

from model_exporter.validation.numeric import _output_names_include_logits


def test_output_names_include_logits_detects_classifier_outputs():
    assert _output_names_include_logits(["logits"]) is True
    assert _output_names_include_logits(["output.logits"]) is True
    assert _output_names_include_logits(["sentence_embedding"]) is False
