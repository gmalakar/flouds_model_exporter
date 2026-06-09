# =============================================================================
# File: test_cli.py
# Date: 2026-04-18
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import textwrap

import pytest


def test_direct_flags_mode_is_rejected(cli_module):
    module, captured = cli_module

    with pytest.raises(SystemExit):
        module.main(
            [
                "--model-name",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--task",
                "feature-extraction",
            ]
        )

    assert captured["calls"] == []


def test_export_subcommand_uses_canonical_cli_names(cli_module, monkeypatch, tmp_path):
    module, captured = cli_module
    onnx_dir = tmp_path / "onnx-output"
    monkeypatch.setenv("ONNX_PATH", str(onnx_dir))

    module.main(
        [
            "export",
            "--model-name",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--task",
            "feature-extraction",
            "--model-for",
            "fe",
            "--opset-version",
            "17",
            "--trust-remote-code",
            "--use-sub-process",
            "--cleanup",
            "--prune-canonical",
        ]
    )

    assert captured["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["task"] == "feature-extraction"
    assert captured["model_for"] == "fe"
    assert captured["opset_version"] == 17
    assert captured["trust_remote_code"] is True
    assert captured["use_subprocess"] is True
    assert captured["cleanup"] is True
    assert captured["prune_canonical"] is True
    assert captured["onnx_path"] == str(onnx_dir)


def test_export_subcommand_forwards_to_export_pipeline(cli_module, monkeypatch, tmp_path):
    module, captured = cli_module
    onnx_dir = tmp_path / "onnx-output"
    monkeypatch.setenv("ONNX_PATH", str(onnx_dir))

    module.main(
        [
            "export",
            "--model-name",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--task",
            "feature-extraction",
            "--model-for",
            "fe",
            "--opset-version",
            "17",
        ]
    )

    assert captured["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["task"] == "feature-extraction"
    assert captured["model_for"] == "fe"
    assert captured["opset_version"] == 17
    assert captured["onnx_path"] == str(onnx_dir)


def test_export_subcommand_forwards_quantize(cli_module):
    module, captured = cli_module

    module.main(
        [
            "export",
            "--model-name",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--task",
            "feature-extraction",
            "--quantize",
            "dynamic_int8",
        ]
    )

    assert captured["quantize"] == "dynamic_int8"


def test_export_subcommand_normalizes_quantize_both(cli_module):
    module, captured = cli_module

    module.main(
        [
            "export",
            "--model-name",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--task",
            "feature-extraction",
            "--quantize",
            "both",
        ]
    )

    assert captured["quantize"] is True


def test_export_subcommand_accepts_optimization_options_when_enabled(cli_module):
    module, captured = cli_module

    module.main(
        [
            "export",
            "--model-name",
            "test-model",
            "--task",
            "feature-extraction",
            "--optimize",
            "--optimization-level",
            "2",
            "--portable",
        ]
    )

    assert captured["optimize"] is True
    assert captured["optimization_level"] == 2
    assert captured["portable"] is True


def test_export_subcommand_accepts_llm_only_options_for_llms(cli_module):
    module, captured = cli_module

    module.main(
        [
            "export",
            "--model-name",
            "test-model",
            "--task",
            "text-generation-with-past",
            "--model-for",
            "llm",
            "--merge",
            "--no-local-prep",
        ]
    )

    assert captured["model_for"] == "llm"
    assert captured["merge"] is True
    assert captured["no_local_prep"] is True


def test_batch_subcommand_runs_recommended_preset(cli_module, monkeypatch, tmp_path):
    module, captured = cli_module
    onnx_dir = tmp_path / "onnx-output"
    monkeypatch.setenv("ONNX_PATH", str(onnx_dir))

    module.main(["batch", "--preset", "recommended", "--min-free-memory-gb", "0"])

    assert len(captured["calls"]) == 2
    assert captured["calls"][0]["model_name"] == "BAAI/bge-base-en-v1.5"
    assert captured["calls"][1]["model_name"] == "cross-encoder/ms-marco-MiniLM-L-12-v2"
    assert captured["calls"][0]["onnx_path"] == str(onnx_dir)


def test_batch_subcommand_loads_custom_config(cli_module, tmp_path):
    module, captured = cli_module
    config_path = tmp_path / "batch-config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            {
              "batch_presets": {
                "custom": [
                  {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "model_for": "fe",
                    "task": "feature-extraction",
                    "library": "transformers",
                    "optimize": true,
                    "optimization_level": 2,
                    "quantize": "fp16"
                  }
                ]
              }
            }
            """
        ).strip(),
        encoding="utf-8",
    )

    module.main(
        [
            "batch",
            "--config",
            str(config_path),
            "--preset",
            "custom",
            "--min-free-memory-gb",
            "0",
        ]
    )

    assert len(captured["calls"]) == 1
    assert captured["calls"][0]["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["calls"][0]["optimize"] is True
    assert captured["calls"][0]["optimization_level"] == 2
    assert captured["calls"][0]["quantize"] == "fp16"


def test_batch_subcommand_applies_optimization_level_override(cli_module, monkeypatch):
    module, captured = cli_module
    monkeypatch.delenv("ONNX_PATH", raising=False)

    module.main(
        [
            "batch",
            "--preset",
            "recommended",
            "--min-free-memory-gb",
            "0",
            "--optimize",
            "--optimization-level",
            "2",
        ]
    )

    assert len(captured["calls"]) == 2
    assert captured["calls"][0]["optimize"] is True
    assert captured["calls"][0]["optimization_level"] == 2
    assert captured["calls"][1]["optimization_level"] == 2


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--optimization-level", "2"],
        ["--portable"],
        ["--prune-canonical"],
        ["--no-local-prep"],
    ],
)
def test_batch_subcommand_rejects_invalid_global_options(cli_module, extra_args):
    module, captured = cli_module

    with pytest.raises(SystemExit):
        module.main(["batch", "--preset", "recommended", *extra_args])

    assert captured["calls"] == []


def test_batch_subcommand_rejects_global_skip_validator_for_validator_entries(
    cli_module,
):
    module, captured = cli_module

    with pytest.raises(SystemExit):
        module.main(
            [
                "batch",
                "--preset",
                "recommended",
                "--skip-validator",
                "--min-free-memory-gb",
                "0",
            ]
        )

    assert captured["calls"] == []


def test_validate_subcommand_forwards_to_validator(cli_module):
    module, captured = cli_module

    rc = module.main(
        [
            "validate",
            "--model-dir",
            "onnx/models/fe/all-MiniLM-L6-v2",
            "--reference-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--device",
            "cpu",
            "--atol",
            "0.001",
            "--rtol",
            "0.01",
            "--normalize-embeddings",
        ]
    )

    assert rc == 0
    assert captured["validate_argv"] == [
        "--model-dir",
        "onnx/models/fe/all-MiniLM-L6-v2",
        "--reference-model",
        "sentence-transformers/all-MiniLM-L6-v2",
        "--device",
        "cpu",
        "--atol",
        "0.001",
        "--rtol",
        "0.01",
        "--normalize-embeddings",
    ]


def test_optimize_subcommand_forwards_to_optimizer(cli_module):
    module, captured = cli_module

    rc = module.main(
        [
            "optimize",
            "--model-dir",
            "onnx/models/fe/all-MiniLM-L6-v2",
            "--model-for",
            "fe",
            "--optimization-level",
            "2",
            "--portable",
        ]
    )

    assert rc == 0
    assert captured["optimize_call"] == {
        "model_dir": "onnx/models/fe/all-MiniLM-L6-v2",
        "model_for": "fe",
        "logger_name": "model_exporter.optimize",
        "optimization_level": 2,
        "portable": True,
    }


@pytest.mark.parametrize(
    "invalid_flag",
    [
        "--model_for",
        "--trust_remote_code",
        "--opset_version",
        "--prune_canonical",
        "--use_sub_process",
        "--pack-single-threshold-mb",
    ],
)
def test_invalid_export_flags_are_rejected(cli_module, invalid_flag):
    module, captured = cli_module

    with pytest.raises(SystemExit):
        module.main(
            [
                "export",
                "--model-name",
                "test-model",
                "--task",
                "feature-extraction",
                invalid_flag,
            ]
        )

    assert captured["calls"] == []


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--portable"],
        ["--optimization-level", "2"],
        ["--prune-canonical"],
        ["--skip-validator", "--require-validator"],
        ["--skip-validator", "--normalize-embeddings"],
        ["--low-memory-env", "--use-external-data-format"],
        ["--low-memory-env", "--no-post-process"],
        ["--no-local-prep"],
        ["--merge"],
    ],
)
def test_invalid_export_option_combinations_are_rejected(cli_module, extra_args):
    module, captured = cli_module

    with pytest.raises(SystemExit):
        module.main(
            [
                "export",
                "--model-name",
                "test-model",
                "--task",
                "feature-extraction",
                *extra_args,
            ]
        )

    assert captured["calls"] == []


@pytest.mark.parametrize(
    "argv",
    [
        [
            "export",
            "--model-name",
            "test-model",
            "--task",
            "feature-extraction",
            "--optimize",
            "--optimization-level",
            "42",
        ],
        [
            "optimize",
            "--model-dir",
            "onnx/models/fe/all-MiniLM-L6-v2",
            "--model-for",
            "fe",
            "--optimization-level",
            "42",
        ],
        [
            "batch",
            "--preset",
            "recommended",
            "--optimization-level",
            "42",
        ],
    ],
)
def test_invalid_optimization_levels_are_rejected(cli_module, argv):
    module, captured = cli_module

    with pytest.raises(SystemExit):
        module.main(argv)

    assert captured["calls"] == []
