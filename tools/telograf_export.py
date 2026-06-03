"""
Subprocess bridge: read a case JSON from stdin, load a TeLoGraF
checkpoint, run flow-matching sampling, dump waypoints as JSON on stdout.

This is the *only* place where we touch TeLoGraF's internals.  Parent
process (`planner.telograf_infer.plan_waypoints`) shells out to this
script in the TeLoGraF venv, so neither side has to know about the other
beyond JSON over stdin/stdout.

PATH LAYOUT EXPECTED
--------------------
After running `bash scripts/install_telograf.sh` plus a checkpoint
download (see planner/README.md), the on-disk layout is:

    code/external/TeLoGraF/
        code/                             # cloned repo source
            train_gstl_v1.py
            z_diffuser.py
            z_models.py
            utils.py
            ...
        exps/                             # downloaded checkpoints
            g0128-075243_simple_gnn_F/    # one experiment per dir
                args.npz                  # training-time argparse Namespace
                models/
                    model_last.ckpt       # the actual state_dict
        exps_data/                        # downloaded datasets (for stats)
            simple/
                stat.npz                  # per-dataset normalisation stats
                ...

The `args.npz` next to the checkpoint is what tells us which encoder
(GNN / GRU / Transformer / TreeLSTM / Goal), which diffuser (Gaussian /
Flow / VAE / Grad), and which hyper-parameters (horizon, data_dim,
condition_dim) to rebuild.  Without it we have to guess and that almost
always ends in shape mismatch errors at load_state_dict.

PROTOCOL
--------
stdin  (one JSON object):
    {
      "id":         "<case_id>",
      "tree":       <STL tree dict>,
      "grounding":  {"prop_1": {...}, ...},
      "start":      [x0, y0],
      "n_steps":    96,
      "node_feats": [[...], ...],          # from planner.case_to_graph
      "edge_index": [[src, dst], ...],
      "ckpt":       "<path to model_last.ckpt>"
    }

stdout (last line is JSON):
    {"waypoints": [[x0, y0], [x1, y1], ...], "backend": "telograf"}

ENV VARS
--------
FAKE_OUTPUT=1   skip the model entirely, return a straight-line dummy
                trajectory.  Used to verify the subprocess plumbing
                works without needing a checkpoint on disk.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
TELOGRAF_CODE = HERE.parent.parent / "external" / "TeLoGraF" / "code"
if str(TELOGRAF_CODE) not in sys.path:
    sys.path.insert(0, str(TELOGRAF_CODE))


def _fail(msg: str, code: int = 2) -> None:
    print(f"telograf_export: {msg}", file=sys.stderr)
    sys.exit(code)


def _reach_goals(case):
    """(x, y, r) for every reach atom in the grounding (kind != avoid)."""
    return [(g["x"], g["y"], g.get("r", 0.5))
            for g in case.get("grounding", {}).values()
            if g.get("kind", "reach") == "reach"]


def _sample_cost(wp, goals, obstacles):
    """Rank a raw sample: reach every goal + clear every obstacle."""
    import math
    cost = 0.0
    for gx, gy, gr in goals:
        cost += max(0.0, min(math.hypot(x - gx, y - gy) for x, y in wp) - gr)
    for ox, oy, orr in obstacles:
        worst = min(math.hypot(x - ox, y - oy) - orr for x, y in wp)
        if worst < 0:
            cost += (-worst) * 10.0
    return cost


def _stl_refine(wp, start, goals, obstacles, *, clear=0.20,
                iters=200, lr=0.02):
    """STLCG-style differentiable refinement (test-time guidance).

    TeLoGraF gives the learned, GNN-conditioned trajectory *shape*; this
    polishes it with a few hundred Adam steps on a differentiable STL /
    obstacle robustness so the FED obstacles are actually cleared and the
    goal is actually reached.  This is the same idea as TeLoGraF's
    CTG/LTLDoG test-time guidance, applied as a light post-process on the
    2-D waypoints (cheap, runs on CPU in <0.1 s).

    Loss = obstacle-violation^2  +  goal-reach^2  +  start-anchor  + smooth.
    """
    import torch

    T = torch.tensor(wp, dtype=torch.float32, requires_grad=True)
    s = torch.tensor(start, dtype=torch.float32)
    opt = torch.optim.Adam([T], lr=lr)
    goal_t = [torch.tensor([gx, gy], dtype=torch.float32) for gx, gy, _ in goals]
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.zeros(())
        # push every waypoint out of every obstacle (inflated by `clear`)
        for ox, oy, orr in obstacles:
            d = torch.sqrt((T[:, 0] - ox) ** 2 + (T[:, 1] - oy) ** 2 + 1e-9)
            loss = loss + 3.0 * (torch.relu(orr + clear - d) ** 2).sum()
        # reach each goal: some part of the trajectory must touch it
        for gt in goal_t:
            loss = loss + 2.0 * ((T - gt) ** 2).sum(dim=1).min()
        # anchor the start, keep the path smooth
        loss = loss + 5.0 * ((T[0] - s) ** 2).sum()
        loss = loss + 0.05 * ((T[1:] - T[:-1]) ** 2).sum()
        loss.backward()
        opt.step()
    return [[float(x), float(y)] for x, y in T.detach().tolist()]


def _fake(case: dict) -> None:
    """FAKE_OUTPUT=1 path - emit a straight-line trajectory so the parent
    can confirm the subprocess pipes work without needing a real model."""
    sx, sy = case["start"]
    n = case["n_steps"]
    # crude straight line of length 1.0 in each axis
    wp = [[sx + (i / n), sy + (i / n)] for i in range(n)]
    print(json.dumps({"waypoints": wp, "samples": [wp],
                      "backend": "telograf-fake"}))


def _load_args_namespace(ckpt_path: Path):
    """Look up the argparse Namespace TeLoGraF wrote at training time.

    TeLoGraF saves it at  <exp_dir>/args.npz  via
        np.savez(os.path.join(args.exp_dir_full, 'args'), args=args)
    """
    import numpy as np

    exp_dir = ckpt_path.parent.parent           # .../models/x.ckpt -> exp_dir
    args_file = exp_dir / "args.npz"
    if not args_file.exists():
        _fail(
            f"args.npz not found at {args_file}.\n"
            "Checkpoint zips on the TeLoGraF Google Drive include this file -- "
            "make sure you unzipped the FULL experiment directory, not just "
            "model_last.ckpt by itself."
        )
    raw = np.load(args_file, allow_pickle=True)
    return raw["args"].item()


def _build_model(args, ckpt_path: Path):
    """Reproduce the encoder + diffuser TeLoGraF would build at the start
    of `train_gstl_v1.main()` for these args, then load the checkpoint.

    Kept deliberately close to lines 1222-1316 of train_gstl_v1.py so a
    side-by-side diff stays easy to do.
    """
    import torch
    from z_diffuser import (
        GaussianDiffusion,
        GaussianFlow,
        GaussianVAE,
        GradNN,
        TemporalUnet,
        MockNet,
        MLPNet,
    )
    from z_models import GCN, GRUEncoder, TransformerModel, TreeLSTMNet, MLP

    # Device: auto-pick CUDA if this torch build actually has it, else CPU.
    # The shipped venv is torch 2.4.1+cpu (no CUDA) so this stays CPU; install
    # a CUDA torch that supports your GPU and it switches automatically.
    # Override with TELOGRAF_DEVICE=cpu|cuda.  NOTE: for this tiny model
    # (GNN + flow, horizon 64, batch ~48) GPU gives little wall-clock benefit --
    # the cost is Python import + the iterative STLCG refinement loop, both
    # CPU-side -- so CPU is a fine default.
    _want = os.environ.get("TELOGRAF_DEVICE", "auto").lower()
    if _want == "cuda" or (_want == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"telograf_export: device={device}", file=sys.stderr)
    input_dim = 8          # we feed the 8-d node features from planner.case_to_graph
    ego_state_dim = args.observation_dim

    if args.encoder == "gnn":
        encoder = GCN(input_dim, ego_state_dim, args)
    elif args.encoder == "gru":
        encoder = GRUEncoder(input_dim, ego_state_dim=ego_state_dim,
                             hidden_dim=args.gru_hidden_dim,
                             feature_dim=args.condition_dim,
                             num_layers=args.gru_num_layers, args=args)
    elif args.encoder == "trans":
        encoder = TransformerModel(input_dim, ego_state_dim, 1000,
                                   args.emsize, args.nhead, args.nhid,
                                   args.nlayers, args.dropout, args)
    elif args.encoder == "tree_lstm":
        encoder = TreeLSTMNet(x_dim=input_dim, ego_state_dim=ego_state_dim,
                              hidden_dim=args.tree_hidden_dim,
                              hidden_layers=args.tree_num_layers, args=args)
    elif args.encoder == "ego":
        encoder = MLP(ego_state_dim, args.condition_dim, args=args)
    elif args.encoder == "goal":
        encoder = GCN(input_dim, ego_state_dim, args)
    elif args.encoder == "zero":
        encoder = None
    else:
        _fail(f"unknown encoder in checkpoint args: {args.encoder!r}")

    if getattr(args, "mock_model", False):
        net = MockNet(args.horizon, args.data_dim,
                      cond_dim=args.condition_dim + ego_state_dim,
                      dim=args.tconv_dim, dim_mults=args.dim_mults,
                      attention=args.attention)
    elif getattr(args, "mlp", False):
        net = MLPNet(args.horizon, args.data_dim,
                     cond_dim=args.condition_dim + ego_state_dim,
                     dim=args.tconv_dim, dim_mults=args.dim_mults,
                     attention=args.attention)
    else:
        net = TemporalUnet(args.horizon, args.data_dim,
                           cond_dim=args.condition_dim + ego_state_dim,
                           dim=args.tconv_dim, dim_mults=args.dim_mults,
                           attention=args.attention,
                           dropout=args.unet_dropout)

    if getattr(args, "grad_nn", False):
        cls = GradNN
    elif getattr(args, "flow", False):
        cls = GaussianFlow
    elif getattr(args, "vae", False):
        cls = GaussianVAE
    else:
        cls = GaussianDiffusion

    diffuser = cls(
        model=net,
        horizon=args.horizon,
        observation_dim=args.data_dim - args.action_dim,
        action_dim=args.action_dim,
        n_timesteps=args.n_timesteps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        clip_value_min=-args.clip_value,
        clip_value_max=args.clip_value,
        encoder=encoder,
        transition_dim=args.data_dim,
    )

    state = torch.load(str(ckpt_path), map_location="cpu")
    diffuser.load_state_dict(state, strict=False)
    diffuser.eval()
    diffuser.to(device)
    return diffuser, encoder, device


def _build_graph_batch(case: dict, device):
    """Turn our (node_feats, edge_index) JSON payload into a
    torch_geometric.data.Batch of size 1 that TeLoGraF's encoder accepts.
    """
    import torch
    from torch_geometric.data import Data, Batch

    x  = torch.tensor(case["node_feats"], dtype=torch.float32)
    ei = (torch.tensor(case["edge_index"], dtype=torch.long).t().contiguous()
          if case["edge_index"] else torch.empty((2, 0), dtype=torch.long))
    return Batch.from_data_list([Data(x=x, edge_index=ei)]).to(device)


def _denorm_xy(traj, exp_dir: Path, args):
    """If the dataset's stat.npz is on disk, denormalise the (x, y)
    channels back to world coordinates.  If it's missing, return the
    trajectory as-is (caller will have to interpret in [-1, 1]^2)."""
    import numpy as np

    # train_gstl_v1.py looks up: <exp_dir>/<data_path>/stat.npz
    stat_path = exp_dir.parent.parent / "exps_data" / args.env / "stat.npz"
    if not stat_path.exists():
        # try a fallback location
        alt = Path(args.data_path) if hasattr(args, "data_path") else None
        if alt and alt.exists():
            stat_path = alt / "stat.npz"
    if not stat_path.exists():
        # No stat.npz on disk -- apply the TeLoGraF "simple" env's training
        # bounds (x, y in [-5, 5], see train_gstl_v1.py line 1342) as a
        # close-enough denormalisation.  The output trajectory is in roughly
        # [-1, 1] normalised space; multiplying by 5 puts it back in world
        # coordinates matched to our case world_bounds (= [-4, 4]).
        SIMPLE_BOUND = 5.0
        xs = (traj[:, 0] * SIMPLE_BOUND).tolist()
        ys = (traj[:, 1] * SIMPLE_BOUND).tolist()
        return list(zip(xs, ys)), False

    stat = np.load(stat_path, allow_pickle=True)["data"].item()
    mu  = np.asarray(stat["obs_mean"])[:2]
    std = np.asarray(stat["obs_std"])[:2]
    xs = traj[:, 0] * std[0] + mu[0]
    ys = traj[:, 1] * std[1] + mu[1]
    return list(zip(xs.tolist(), ys.tolist())), True


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        _fail("no input on stdin")
    try:
        case = json.loads(raw)
    except json.JSONDecodeError as e:
        _fail(f"invalid JSON on stdin: {e}")

    if os.environ.get("FAKE_OUTPUT") == "1":
        return _fake(case)

    ckpt = case.get("ckpt")
    if not ckpt or not os.path.exists(ckpt):
        _fail(f"checkpoint not found: {ckpt}")
    ckpt_path = Path(ckpt).resolve()

    try:
        import torch                                   # noqa: F401
        from z_diffuser import GaussianFlow             # noqa: F401
        import z_models                                  # noqa: F401
        import torch_geometric                           # noqa: F401
    except ImportError as e:
        _fail(f"TeLoGraF import failed (is the venv active?): {e}", code=3)

    args = _load_args_namespace(ckpt_path)

    # The training-time argparse Namespace may be missing attributes that
    # newer TeLoGraF versions added (flow_pattern, guidance flags, ...).
    # The sampling path branches on these, so default them to safe values.
    for attr, default in [
        ("flow_pattern",       None),
        ("guidance_before",    None),
        ("guidance_steps",     0),
        ("cls_guidance",       False),
        ("guidance_data",      None),
        ("smoothing_factor",   None),
        ("guidance_lr",        0.0),
        ("guidance_scale",     0.0),
        ("guidance",           False),
        ("rl",                 False),
        ("env",                "simple"),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    diffuser, encoder, device = _build_model(args, ckpt_path)
    batch = _build_graph_batch(case, device)

    import numpy as np
    import torch

    # ----- best-of-N batch sampling -------------------------------------
    # TeLoGraF is used with `test_muls` (the paper samples up to 1024
    # trajectories per STL and keeps the best by STL robustness).  A single
    # flow sample is noisy; a batch lets the caller select a satisfying one.
    # We sample `n_samples` in ONE forward (the diffuser handles batches),
    # so there's still only one checkpoint load per subprocess call.
    n_samples = int(case.get("n_samples", 1))

    # TeLoGraF's encoders use signature  forward(ego_states, data).
    # ego_states is [B, observation_dim], data is the torch_geometric Batch.
    if encoder is not None:
        sx, sy = case["start"]
        ego = torch.tensor([[float(sx), float(sy)]], dtype=torch.float32,
                            device=device)
        with torch.no_grad():
            cond_emb = encoder(ego, batch)
        if isinstance(cond_emb, (tuple, list)):
            cond_emb = cond_emb[0]
    else:
        cond_emb = torch.zeros(1, args.condition_dim + args.observation_dim,
                               device=device)

    # Repeat the conditioning to a batch of n_samples.
    cond_batch = cond_emb.repeat(n_samples, 1)

    # GaussianFlow.p_sample_loop wants (shape, cond, args=args).  Note: the
    # upstream `in_painting` kwarg is only a debug-print flag in the flow
    # branch (z_diffuser.py:785-790) -- `apply_conditioning` is NOT called
    # in flow inference, so samples come from the prior and we re-anchor
    # the start post-hoc per sample.
    shape = (n_samples, args.horizon, args.data_dim)
    sample_kwargs = {"args": args}
    with torch.no_grad():
        try:
            x_traj = diffuser.conditional_sample(cond_batch,
                                                 horizon=args.horizon,
                                                 **sample_kwargs)
        except TypeError:
            x_traj = diffuser.p_sample_loop(shape, cond_batch, args=args)

    if hasattr(x_traj, "trajectories"):
        traj_t = x_traj.trajectories
    elif isinstance(x_traj, (tuple, list)):
        traj_t = x_traj[0]
    else:
        traj_t = x_traj                                # [N, H, D]

    n_target = case["n_steps"]
    do_shift = os.environ.get("TELOGRAF_NO_SHIFT") != "1"
    sx, sy = case["start"]

    all_wps = []
    denormalised = False
    for s in range(traj_t.shape[0]):
        traj_np = traj_t[s].detach().cpu().numpy()     # [H, D]
        wp_xy, denormalised = _denorm_xy(traj_np, ckpt_path.parent.parent, args)
        if do_shift:
            dx, dy = sx - wp_xy[0][0], sy - wp_xy[0][1]
            wp_xy = [(x + dx, y + dy) for (x, y) in wp_xy]
        if len(wp_xy) >= n_target:
            step = max(1, len(wp_xy) // n_target)
            wp_xy = wp_xy[::step][:n_target]
        all_wps.append([[float(x), float(y)] for (x, y) in wp_xy])

    if not denormalised:
        print(f"telograf_export: WARN: no stat.npz found, using {args.env!r} "
              f"env's training bounds (~5.0 m) for denormalisation",
              file=sys.stderr)

    # ----- best-of-N selection + STLCG-style obstacle refinement ---------
    # `refine_obstacles` is the FULL obstacle set the caller wants honoured
    # (STL avoid atoms + procedural walls/furniture-as-circles).  This is
    # how the procedural obstacles get *into* the diffusion planner: the
    # GNN gives the shape, the differentiable guidance enforces them.
    obstacles = [(o["x"], o["y"], o["r"])
                 for o in case.get("refine_obstacles", [])]
    goals = _reach_goals(case)
    start = (float(sx), float(sy))

    best = min(all_wps, key=lambda w: _sample_cost(w, goals, obstacles))
    refined = best
    if obstacles or goals:
        try:
            refined = _stl_refine(best, start, goals, obstacles)
            print(f"telograf_export: refined best-of-{len(all_wps)} against "
                  f"{len(obstacles)} obstacle(s) + {len(goals)} goal(s)",
                  file=sys.stderr)
        except Exception as e:                          # never crash the demo
            print(f"telograf_export: refine skipped ({e})", file=sys.stderr)

    print(json.dumps({"waypoints": refined,
                      "samples":   all_wps,
                      "best_raw":  best,
                      "backend":   "telograf",
                      "denormalised": denormalised,
                      "encoder":   args.encoder,
                      "horizon":   args.horizon}))


if __name__ == "__main__":
    main()
