#!/usr/bin/env bash
# Install TeLoGraF (https://github.com/mengyuest/TeLoGraF) into:
#
#     code/external/TeLoGraF/        <- repo source
#     code/.venv-telograf/           <- isolated venv with matching deps
#
# This script is idempotent: rerun it after a `git pull` or to verify the
# install.  It does NOT modify the system Python install, ~/.bashrc, or
# any ROS package.
#
# We deliberately use a venv (not conda).  TeLoGraF upstream pins
# pytorch==1.13.1 + CUDA 11.7 which (a) does not have an RTX 50-series
# build and (b) ties us to old conda channels.  We use CPU-only torch
# 2.4 here -- TeLoGraF inference is graph-tiny (one STL tree, <30
# nodes) and runs in well under a second on CPU.
#
# If you have a 4090/A100 or older Ampere GPU you want to use, run:
#     PYTORCH_CHANNEL=cu121 bash scripts/install_telograf.sh
# and it will switch to the GPU wheel.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$HERE/.." && pwd)"

REPO_URL="${TELOGRAF_REPO:-https://github.com/mengyuest/TeLoGraF.git}"
TELOGRAF_DIR="$CODE_ROOT/external/TeLoGraF"
VENV_DIR="$CODE_ROOT/.venv-telograf"
PYTORCH_CHANNEL="${PYTORCH_CHANNEL:-cpu}"     # cpu | cu118 | cu121 | cu124

# 1) Sanity checks --------------------------------------------------------
if ! command -v python3 >/dev/null; then
    echo "ERROR: python3 not found." >&2; exit 1
fi
PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VERSION" != "3.10" && "$PY_VERSION" != "3.11" ]]; then
    echo "WARN: python3 is $PY_VERSION; TeLoGraF is tested with 3.10."
    echo "      Continuing -- if torch_geometric fails to install, install Python 3.10."
fi

if ! command -v git >/dev/null; then
    echo "ERROR: git not found.  sudo apt-get install -y git" >&2; exit 1
fi

# 2) Clone or update the TeLoGraF repo ------------------------------------
mkdir -p "$CODE_ROOT/external"
if [[ -d "$TELOGRAF_DIR/.git" ]]; then
    echo "[1/4] TeLoGraF already cloned at $TELOGRAF_DIR -- skipping clone"
else
    echo "[1/4] Cloning TeLoGraF from $REPO_URL ..."
    git clone --depth 1 "$REPO_URL" "$TELOGRAF_DIR"
fi

# 3) Build the venv -------------------------------------------------------
if [[ -x "$VENV_DIR/bin/python3" ]]; then
    echo "[2/4] venv already exists at $VENV_DIR -- reusing"
else
    echo "[2/4] Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools >/dev/null

# 4) Install matching deps ------------------------------------------------
echo "[3/4] Installing PyTorch (${PYTORCH_CHANNEL}) ..."
case "$PYTORCH_CHANNEL" in
    cpu)
        python -m pip install --index-url https://download.pytorch.org/whl/cpu \
            torch==2.4.1 torchvision==0.19.1
        ;;
    cu118|cu121|cu124)
        python -m pip install --index-url "https://download.pytorch.org/whl/${PYTORCH_CHANNEL}" \
            torch==2.4.1 torchvision==0.19.1
        ;;
    *)
        echo "ERROR: unknown PYTORCH_CHANNEL=$PYTORCH_CHANNEL (use cpu|cu118|cu121|cu124)" >&2
        exit 1
        ;;
esac

echo "    Installing PyTorch Geometric ..."
# torch-scatter / torch-sparse have GPU-specific wheels; for CPU and small
# graphs we use the plain CPU build that comes with pyg-lib turned off.
python -m pip install torch_geometric==2.5.3
python -m pip install \
    --no-cache-dir \
    -f "https://data.pyg.org/whl/torch-2.4.0+${PYTORCH_CHANNEL}.html" \
    torch-scatter torch-sparse || \
    echo "    NOTE: torch-scatter/torch-sparse failed -- only needed for some PyG layers; continuing."

echo "    Installing TeLoGraF Python deps ..."
python -m pip install \
    networkx einops pandas matplotlib imageio \
    pyyaml tqdm scipy

# Optional deps (only needed for some TeLoGraF eval routines, not core inference).
python -m pip install cma || echo "    NOTE: cma optional; skipping."
python -m pip install gymnasium pybullet pytorch_kinematics \
    || echo "    NOTE: gymnasium/pybullet stack optional; skipping."

# Gurobi: TeLoGraF's CTG/LTLDoG baselines use it.  Install via pip only;
# the user still has to set GRB_LICENSE_FILE.  Skipping is safe for the
# flow-matching planner path that the poster demo uses.
python -m pip install gurobipy || echo "    NOTE: gurobipy optional; skipping."

# 5) Verify ---------------------------------------------------------------
echo "[4/4] Verifying ..."
python -c "
import torch, torch_geometric, sys
print('   torch          ', torch.__version__)
print('   torch_geometric', torch_geometric.__version__)
print('   cuda available ', torch.cuda.is_available())
print('   python         ', sys.version.split()[0])
"

# Project-side smoke test (graph builder, not the diffusion model itself,
# so we don't need a checkpoint to verify the wiring works).
PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" \
    python -c "
from planner import CASES, case_to_graph, plan_waypoints, telograf_available
print()
print('   telograf_available =', telograf_available())
for c in CASES:
    g  = case_to_graph(c)
    wp = plan_waypoints(c, backend='fallback', n_steps=32)
    print(f'   {c[\"id\"]:18s}  nodes={len(g.node_features):2d}  '
          f'fallback_waypoints={len(wp)}')
"

cat <<EOF

============================================================
TeLoGraF install complete.

Next steps:
  1. Download a checkpoint (optional, only if you want the real
     diffusion planner instead of the robustness fallback):
         see https://github.com/mengyuest/TeLoGraF#models and put
         model_last.ckpt under code/external/TeLoGraF/exps/<run>/models/

  2. Pick a backend at run time:
         backend=fallback   (default if no ckpt found)
         backend=telograf   (only if ckpt exists)
         backend=auto       (use telograf if available)

  3. Activate the venv when you want to call TeLoGraF from a shell:
         source "$VENV_DIR/bin/activate"

The ROS launch files (sim_ros2/launch/*.launch.py) do NOT need the venv
sourced -- they use the system python where ROS lives.  The TeLoGraF
import inside planner.telograf_infer.plan_waypoints will fall back to
the robustness planner if torch is missing from the active interpreter.
============================================================
EOF
