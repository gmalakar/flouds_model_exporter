# Security Policy

## Supported Versions

This project is in active development. Security fixes are applied to the latest `main` branch and the latest published release when a release exists.

| Release | Status |
| --- | --- |
| 2.x | Supported |
| 1.x | Not supported; upgrade to 2.x (see the changelog's Security section) |

| Runtime | Status |
| --- | --- |
| Python 3.11 | Supported |
| Python 3.12 | Supported |
| Other Python versions | Not supported |

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Instead, report privately by contacting the maintainer:

- Contact: Goutam Malakar
- Preferred channel: GitHub private vulnerability reporting (Security Advisories)

When reporting, include:

- A clear description of the vulnerability
- Impact and affected components
- Reproduction steps or proof-of-concept
- Suggested mitigation, if available

## Response process

- Acknowledgement target: within 5 business days
- Triage and severity assessment after acknowledgement
- Fix planning and coordinated disclosure timeline

## Security Notes

- Do not publish Hugging Face tokens, ONNX artifacts containing private model data, or validation dumps from private models.
- `--trust-remote-code` executes model repository code. Enable it only after reviewing the model repository and source. The exporter never enables it by itself: a model that needs it fails with `RemoteCodeNotAllowedError` until you pass the flag.
- A token given to `export()` or `--huggingface_hub_token` is used for that export only. It is not written to disk, no login is performed, and it is not left in the environment. Prefer the `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN` environment variable on shared machines, because command-line arguments are visible in the process list.
- Model names and revisions are validated before use; values that a command line would read as options are refused.
- Use `--revision <commit>` to export a fixed commit of a model repository.
- If a token is exposed, revoke it immediately and create a replacement token.

## Disclosure

Please allow time for a fix before public disclosure.
