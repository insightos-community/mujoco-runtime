# Isolated runtimes for the MuJoCo family

[English](README.md) | [简体中文](README.zh-CN.md)

This directory holds the standalone run configurations for robosuite, LIBERO,
and LIBERO-Pro. They share the test report format, but they are not installed
into the native MuJoCo Runtime's Python environment.

## Environment boundaries

- The native Runtime uses the MuJoCo and R1 Pro assets required by the product.
- robosuite uses Python 3.10, robosuite 1.5.2, and MuJoCo 3.4.0.
- LIBERO uses Python 3.8, robosuite 1.4.0, and NumPy 1.22.4.
- LIBERO-Pro reuses the LIBERO Runtime, adding only pinned-source BDDL
  perturbations and a comparison report.

env.step() and action arrays exist only inside the profiles and are not
exposed to the Agent, Robot Skill, or Studio.

## Common tests

    make test-profiles

The common tests only verify the runner, reports, PNG evidence, and error
handling; they cannot replace real-environment smoke tests.

## robosuite

Install the isolated environment:

    uv sync --project profiles/robosuite --python 3.10 --frozen

Run Lift and Stack:

    MUJOCO_GL=egl uv run --project profiles/robosuite --frozen \
      semantic-sim-profile robosuite --environment Lift --seed 7 --steps 10 \
      --output-dir .output/profiles/robosuite-lift

    MUJOCO_GL=egl uv run --project profiles/robosuite --frozen \
      semantic-sim-profile robosuite --environment Stack --seed 7 --steps 10 \
      --output-dir .output/profiles/robosuite-stack

When EGL is unavailable, MUJOCO_GL can be changed to osmesa. The report
includes Observation fields, reward, success, an RGB PNG, and a 16-bit Depth
PNG. A neutral-control smoke run does not mean the task has been completed.

## LIBERO source preparation

The upstream LIBERO regular wheel does not contain the libero source at the
pinned commit, so this project does not install that empty wheel. Check out
the two repositories in sources.lock.yaml at the pinned commits; the source
root must be passed explicitly at runtime.

    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git .output/sources/LIBERO
    git -C .output/sources/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
    git clone https://github.com/RLinf/LIBERO-PRO.git .output/sources/LIBERO-PRO
    git -C .output/sources/LIBERO-PRO checkout 0bcf73621c789ffd6ed8858467a89df9ca94fd6b

Install the Python 3.8 isolated environment:

    uv sync --project profiles/libero --python 3.8 --frozen

Run a pinned LIBERO task/init-state:

    MUJOCO_GL=egl uv run --project profiles/libero --frozen \
      semantic-sim-profile libero --libero-root .output/sources/LIBERO \
      --suite libero_spatial --task-id 0 --init-state-id 0 --seed 7 --steps 10 \
      --output-dir .output/profiles/libero

Run the LIBERO-Pro base/perturbation comparison:

    MUJOCO_GL=egl uv run --project profiles/libero --frozen \
      semantic-sim-profile libero-pro \
      --libero-root .output/sources/LIBERO \
      --libero-pro-root .output/sources/LIBERO-PRO \
      --evaluation-config profiles/libero/libero-pro.example.yaml \
      --perturbation environment \
      --suite libero_spatial --task-id 0 --init-state-id 0 --seed 7 --steps 10 \
      --output-dir .output/profiles/libero-pro

`sources.lock.yaml` is the single source of upstream versions for release
builds and pinned-baseline tests; when upgrading, update that lock file. The
build tools verify the actual source commit and write the source version into
the scene package. The checkout values above show the current baseline. The
runtime code does not pin a commit, nor does it judge compatibility by commit
equality: source mode records the actual Git HEAD, scene-package mode records
the in-package `source_revision`, and source archives without Git information
remain unknown. Compatible environments are declared by the Runtime Profile,
and the necessary interfaces and assets are still checked at load time;
switching versions requires re-verification. The LIBERO configuration is
written into the current output directory and does not modify the user's home
directory. LIBERO-Pro's Python 3.10 annotations are lazily resolved through a
loading compatibility layer, leaving the upstream source itself unchanged.

## Outputs and verdicts

- report.json: the pinned environment, seed, step count, language objective,
  reward, success, and dependency versions.
- comparison.json: the LIBERO-Pro base and perturbation results and their
  differences.
- initial/final RGB PNG: the acceptance frames.
- initial/final Depth PNG: the normalized 16-bit depth evidence.
- On runtime errors, report.json is still written and the CLI returns a
  non-zero exit code.

External datasets do not enter the Plugin artifacts. Product wiring for the
Framework, Studio, Pilot, AbilityFramework, and Semantic Map is done on their
respective release branches.
