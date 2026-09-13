# mujoco-runtime: reproducible platform builds

## Versions, tools and build layout

The glibc installer pins component tag **v0.5.0-insightos.2026.2** at `03513d48fc6427d3205d63cd29a80ab8dc16b554`.
This guide pins the current build-script snapshot at `d720d276ea23262a50bc0ddbc44ae50a5ffab434`.
To reconstruct another published release, read its `release.json` and select
both `source_commit` and `build_recipe_commit`; a source tag alone may predate
the CI scripts. This recipe reproduces the build steps, not historical archive bytes.

Prerequisites: Linux x86_64, Git, Make, uv 0.12.12, Python build tooling, zstd and PyYAML. The glibc release script selects Python 3.10.19 and the committed uv.lock.

The release scripts expect **two sibling checkouts**, `automation/` for build
scripts and `source/` for the component. Run these commands from a fresh working
directory (the scripts themselves are not standalone copies):

```bash
REPRO_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mujoco-runtime-repro.XXXXXXXX")"
git clone --no-checkout https://github.com/insightos-community/mujoco-runtime.git "$REPRO_ROOT/automation"
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPRO_ROOT/automation" checkout --detach d720d276ea23262a50bc0ddbc44ae50a5ffab434
git clone --no-checkout https://github.com/insightos-community/mujoco-runtime.git "$REPRO_ROOT/source"
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPRO_ROOT/source" checkout --detach v0.5.0-insightos.2026.2
cd "$REPRO_ROOT/source"
test "$(git rev-parse HEAD)" = 03513d48fc6427d3205d63cd29a80ab8dc16b554
export TARGET_TAG=v0.5.0-insightos.2026.2
export COMPONENT=mujoco-runtime
export GITHUB_SHA=d720d276ea23262a50bc0ddbc44ae50a5ffab434
```

## Linux glibc / standard component Release

The executable build entry is [`.github/scripts/build.sh`](.github/scripts/build.sh);
archive validation is [`.github/scripts/package.py`](.github/scripts/package.py).
The runtime pack also reads the Framework and scene catalogs. Reproduce the two
additional dependency checkouts from the CI workflow before invoking the build;
their directory names are part of the build-script interface. Only catalog metadata
is consumed here, so the asset checkout does not need an LFS payload download.

```bash
mkdir -p "$REPRO_ROOT/dependencies/semantic-scene"
git clone --no-checkout https://github.com/insightos-community/Semantic-Framework.git "$REPRO_ROOT/dependencies/semantic-framework"
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPRO_ROOT/dependencies/semantic-framework" checkout --detach 81ea016480f099f9db95bd48c083ee99d8806a4d
git clone --no-checkout https://github.com/insightos-community/mujoco-asset.git "$REPRO_ROOT/dependencies/semantic-scene/mujoco-asset"
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPRO_ROOT/dependencies/semantic-scene/mujoco-asset" checkout --detach f9855e6dd1419f890b418a9f398bd1d5ec49b57c
```

From `source/` in the layout above:

```bash
bash ../automation/.github/scripts/build.sh
python3 ../automation/.github/scripts/package.py 
(cd .output/release && sha256sum -c SHA256SUMS)
```

Artifacts: `source/.output/release/` (archives/wheels, `release.json`, checksum
inventory and license notices). `release.json` records source and recipe revisions.
The local commands do not publish or overwrite a GitHub Release.

## Linux musl

The Linux wheelhouse/runtime pack must be replaced, not copied into musl.
The complete musl assembly is maintained in quick-start; it pins the Python 3.13
Runtime adaptation in `artifacts/musl/upstream.json` and compiles the application
wheels against the verified musl dependency releases. It does not run this
repository’s Python 3.10 Linux release command unchanged.

For the complete musl build and offline checks, use the [quick-start musl commands](https://github.com/insightos-community/quick-start/blob/main/README.build.md#linux-musl-x86_64).

## macOS / macosx

Use Apple Silicon arm64 and the native adaptation at `d720d276ea23262a50bc0ddbc44ae50a5ffab434`;
the older glibc component tag above may not contain the macOS fixes. Start a
separate checkout and run the native commands:

```bash
git clone https://github.com/insightos-community/mujoco-runtime.git mujoco-runtime-macos
cd mujoco-runtime-macos
git checkout --detach d720d276ea23262a50bc0ddbc44ae50a5ffab434
test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64
```

Prerequisites: uv 0.12.12 and the managed Python 3.13.15.

```bash
uv sync --frozen --python 3.13.15 --extra dev
MUJOCO_GL=cgl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest -p pytest_cov -m 'not native'
MUJOCO_GL=cgl uv run --frozen python tools/macos_smoke.py --physics-only --output .output/macos-report.json
uv build
uv build --project packages/mujoco-visuals --out-dir dist
```

On a physical Mac with a working graphics session, omit `--physics-only` for
CGL validation. Hosted-runner physics success does not establish RGB/depth or
GPU performance. The optional `graphics` workflow input requires a registered
`semantic-graphics` self-hosted Mac and runs the pinned pallet scene assets.

The native workflow is [`.github/workflows/macos.yml`](.github/workflows/macos.yml).
Its artifacts are component development outputs; quick-start assembles and validates
the complete installer.

The complete macOS installer targets Apple Silicon/macOS 15.5+; see the [locked assembly instructions](https://github.com/insightos-community/quick-start/blob/main/README.build.md#macos-apple-silicon).

## GitHub workflow reproduction

The repository’s [CI workflow](.github/workflows/ci.yml) implements the two-checkout
layout. To build a source tag without publishing, create a reproduction branch at the
pinned automation commit. GitHub dispatch expects a branch/tag ref; both tag refs
and default-branch dispatches can enter this workflow’s publishing job. The following
commands require repository write access and use a non-default branch:

```bash
gh auth setup-git
REPRO_BRANCH=reproduce/platform-builds
git -C "$REPRO_ROOT/automation" push origin d720d276ea23262a50bc0ddbc44ae50a5ffab434:refs/heads/$REPRO_BRANCH
gh workflow run ci.yml --repo insightos-community/mujoco-runtime --ref "$REPRO_BRANCH" -f tag=v0.5.0-insightos.2026.2
gh run list --repo insightos-community/mujoco-runtime --workflow ci.yml --limit 5
# Set REPRO_RUN_ID to the selected run ID.
gh run watch "$REPRO_RUN_ID" --repo insightos-community/mujoco-runtime --exit-status
gh run download "$REPRO_RUN_ID" --repo insightos-community/mujoco-runtime --name release-assets --dir downloaded-release
```

```bash
git -C "$REPRO_ROOT/automation" push origin d720d276ea23262a50bc0ddbc44ae50a5ffab434:refs/heads/reproduce/macos
gh workflow run macos.yml --repo insightos-community/mujoco-runtime --ref reproduce/macos
```

## Reproduction evidence

Build in a fresh checkout and a separate output directory for each ABI. Preserve
source commits, compiler/tool versions, dependency locks, package inventories and
test logs. Fixed source revisions and a container digest reproduce the recipe;
unlocked OS packages, runner images, timestamps and build tools can still change
archive bytes. Compare a downloaded release against its published `SHA256SUMS`;
do not expect a local rebuild to have the same digest.

See the [complete installer and repository index](https://github.com/insightos-community/quick-start/blob/main/README.build.md) for assembly order,
platform locks and end-to-end validation. Local build commands do not publish a
Release. Publishing requires repository write access and a new version tag;
existing release tags/assets should not be replaced.
