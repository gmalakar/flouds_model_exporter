import textwrap

import pytest



def test_direct_flags_mode_uses_canonical_cli_names(cli_module, monkeypatch, tmp_path):
    module, captured = cli_module
    onnx_dir = tmp_path / "onnx-output"
    monkeypatch.setenv("ONNX_PATH", str(onnx_dir))

    module.main(
        [
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
            "--prune-canonical",
        ]
    )

    assert captured["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["task"] == "feature-extraction"
    assert captured["model_for"] == "fe"
    assert captured["opset_version"] == 17
    assert captured["trust_remote_code"] is True
    assert captured["use_subprocess"] is True
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
                    "optimize": true
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
    ],
)
def test_underscore_flags_are_rejected(cli_module, invalid_flag):
    module, captured = cli_module

    with pytest.raises(SystemExit):
        module.main(
            [
                "--model-name",
                "test-model",
                "--task",
                "feature-extraction",
                invalid_flag,
            ]
        )

    assert captured["calls"] == []
