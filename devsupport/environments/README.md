# Runtime versions and clean core reproduction

Captured 2026-10-09. These are installed-version records and exact version pins,
not wheel-hash locks. The upstream application also retains its original
`uv.lock` and frontend lockfile. GPU inference remains in the existing WSL
environment; none of these records establish a clean reinstall of the native
DocsGPT application or GPU environment.

The original `windows-native.json` remains the historical 2026-10-09 capture.
On 2026-10-10 the unified native evaluation runner required three previously
missing distributions: pyarrow 24.0.0, bm25s 0.2.14 and scipy 1.18.1. These were
added to the existing native environment with no changes to any previously
installed version. The complete after-inventory and comparison are recorded in
`reports/native-evaluation-dependencies-verification-20261010-v1.json`; the exact
existing-environment command and prerequisites are in `docs/RUNBOOK.md`.
The actual fixed DEV12 runner completed with zero operational errors and paid
generation calls. This supplements the historical inventory; it does not assert
an independent clean application reinstall or numerical runtime equality with
the Python 3.10 core environment.

| File | What it records |
|---|---|
| `windows-native.json` | Actual task-owned Python 3.12.15 application environment |
| `wsl-models.json` | Actual Python 3.10.12 GPU environment, Torch 2.11.0+cu128 |
| `windows-core-clean.json` | Independent Python 3.10.11 core environment, 58 distributions, no global package inheritance |
| `windows-core-clean.txt` | Exact version pins from that independently verified environment |

The original core environment inherited global Python packages and had multiple
visible Pillow/setuptools versions. It remains unchanged; its global package
inventory is not distributed. Historical DEV generation-v2 used that environment.
After that batch ended, fresh adaptive/protocol experiments explicitly use
`F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe`. Historical results
are not relabelled as having used the clean environment.

To reproduce the clean core environment, choose a **new** directory. From the
full repository checkout, using a locally available CPython 3.10 and `uv`:

```powershell
uv venv --python C:/Python310/python.exe F:/path/to/new-core-environment
uv pip install --python F:/path/to/new-core-environment/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
uv pip install --python F:/path/to/new-core-environment/Scripts/python.exe -r devsupport/environments/windows-core-clean.txt
uv pip check --python F:/path/to/new-core-environment/Scripts/python.exe
& F:/path/to/new-core-environment/Scripts/python.exe -m pytest -c devsupport/pyproject.toml devsupport/tests -q
```

CPU Torch is only needed here for offline tensor/regression tests. This
environment sends actual embedding and reranking requests to the separately
verified WSL GPU service. Do not use it as a CUDA model host. Browser-form tests
also require Node.js; the verified run used the already installed Node runtime.

Evidence: 310 core tests passed in this clean environment, with four upstream
deprecation warnings (Starlette/AnyIO and FAISS SWIG). Dependency compatibility
check passed. Ten DEV control questions were rerun using the same full contextual
index and existing GPU service: ordered prepack IDs, packed IDs, context text
hashes, and both metric sets matched the original core environment exactly.
No paid calls and no TEST evaluation were made for this check. The scoped record
is `reports/core-clean-reproduction-controls/environment-parity.json`.

At main commit `9fa32da`, the complete current core suite passed **578 tests**
with the same four dependency deprecation warnings. The installed inventory was
also rechecked: all 58 recorded package versions match, with no added, missing
or changed distribution. Evidence is in
`reports/integrated-core-suite-20261009-0534.json` and
`reports/core-version-parity-20261009-0537.json`. These are code and installed-version
checks, not new native/GPU installation or benchmark quality acceptance.

These results demonstrate the tested core path's reproduction on this machine;
they do not establish GPU bitwise portability, another operating system's
compatibility, or complete application installation from scratch.

The hash-verified `docsgpt/core/encodings/cl100k_base.tiktoken` is marked `-text`
in `.gitattributes` so Git preserves its exact bytes on Windows. A fresh local
clone checked out with `core.autocrlf=true` reproduced the expected SHA256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.
This addresses a separate checkout failure observed in an isolated worktree;
it does not relax the runtime checksum. Evidence is recorded in
`reports/tokenizer-checkout-verification-20261009.json`.

## Optional offline plots and media

The current plot and recorded-browser-video exports used the existing Python
3.10.11 core environment at
`F:/DevSupportCopilotWork/.venv-core/Scripts/python.exe`, which already has the
graphics dependencies. The isolated `core-repro` environment does not contain
matplotlib; do not assume a successful core test run verifies plotting, or
install new packages into it during a frozen evaluation.

`devsupport/requirements-analysis.txt` records the five direct graphics/media
versions actually used: numpy 2.2.6, matplotlib 3.10.7, Pillow 12.2.0, imageio
2.37.3 and imageio-ffmpeg 0.6.0. For another machine, create a separate optional
analysis environment and install that file. These are version pins rather than
wheel-hash locks or evidence of a clean installation on another machine.
The imageio-ffmpeg package supplies the encoder; no Windows UI automation or
screen capture is performed by FFmpeg. The recording's raw frames were obtained
only through the supported Cua IAB interface. Its largest sampling gap is
disclosed in the video-verification artifact.

After the UI budget guard and identifier-capacity preflight integrations, the
current core suite passed 660 tests with four warnings at commit `dca9d53`;
`reports/runtime-integration-20261009-1536.json` records that verification.
Later documentation, offline plots and media do not establish benchmark quality,
human validation or final TEST acceptance.
