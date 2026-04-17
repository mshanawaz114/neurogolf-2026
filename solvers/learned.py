from __future__ import annotations

"""
learned.py — Batched training fallback solver for NeuroGolf 2026.

Two-stage strategy:
  Stage 1: Train on TRAIN pairs only (3-5 examples, fast).
           If pixel-perfect AND generalises to ALL → done.
  Stage 2: Retrain on ALL pairs (train + test + arc-gen).

Architecture search: smallest-first.
  1×1 conv → 3×3 conv → 2-layer → 3-layer ...

Training: Adam, cross-entropy, early-stop when pixel-perfect OR stagnation.

Backend: tries PyTorch first (fast), falls back to NumPy (no CUDA needed).
"""

from pathlib import Path
import numpy as np
import onnx

from solvers.base import BaseSolver
from utils.arc_utils import grid_to_tensor, grid_to_array, CANVAS, C
from utils.onnx_builder import save as onnx_save


# ── Try to import PyTorch ──────────────────────────────────────────────────────

def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        _ = torch.zeros(2, 2)  # sanity check
        return torch, nn, optim
    except Exception:
        return None, None, None


# ── PyTorch models ────────────────────────────────────────────────────────────

class _CoordResNet:
    def __init__(self, hidden, kernel, depth, torch, nn, use_coords=False, residual=False):
        self.torch = torch
        self.use_coords = use_coords
        pad = kernel // 2

        class _Net(nn.Module):
            def __init__(self, hidden, kernel, depth, use_coords, residual):
                super().__init__()
                in_c = C + (2 if use_coords else 0)
                self.use_coords = use_coords
                pad = kernel // 2

                class _ResidualModule(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.block = nn.Sequential(
                            nn.Conv2d(hidden, hidden, kernel, padding=pad),
                            nn.ReLU(),
                            nn.Conv2d(hidden, hidden, kernel, padding=pad),
                        )
                        self.relu = nn.ReLU()

                    def forward(self, x):
                        return self.relu(x + self.block(x))

                if use_coords:
                    ys = torch.linspace(-1.0, 1.0, CANVAS, dtype=torch.float32).view(1, 1, CANVAS, 1).expand(1, 1, CANVAS, CANVAS)
                    xs = torch.linspace(-1.0, 1.0, CANVAS, dtype=torch.float32).view(1, 1, 1, CANVAS).expand(1, 1, CANVAS, CANVAS)
                    self.register_buffer("coord_y", ys)
                    self.register_buffer("coord_x", xs)

                layers = []
                if residual:
                    layers.append(nn.Conv2d(in_c, hidden, kernel, padding=pad))
                    layers.append(nn.ReLU())
                    for _ in range(max(depth - 2, 1)):
                        layers.append(_ResidualModule())
                    layers.append(nn.Conv2d(hidden, C, kernel, padding=pad))
                else:
                    cur_c = in_c
                    for _ in range(depth - 1):
                        layers += [nn.Conv2d(cur_c, hidden, kernel, padding=pad), nn.ReLU()]
                        cur_c = hidden
                    layers.append(nn.Conv2d(cur_c, C, kernel, padding=pad))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                if self.use_coords:
                    n = x.shape[0]
                    x = torch.cat(
                        [x, self.coord_y.expand(n, -1, -1, -1), self.coord_x.expand(n, -1, -1, -1)],
                        dim=1,
                    )
                return self.net(x)

        self.net = _Net(hidden, kernel, depth, use_coords, residual)

    def __call__(self, x):
        return self.net(x)
    def parameters(self):
        return self.net.parameters()
    def train(self):
        self.net.train()
    def eval(self):
        self.net.eval()


def _train_torch(model, X, Y, max_epochs, lr, torch, optim, nn,
                 stagnation_patience=80, deadline=None):
    """Train with stagnation early-stop and optional wall-clock deadline."""
    import time
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    sched     = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5, min_lr=1e-5)
    model.train()
    best_loss = float("inf")
    stall = 0
    for epoch in range(max_epochs):
        if deadline is not None and time.time() > deadline:
            return False   # time limit exceeded
        optimizer.zero_grad()
        out  = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()
        sched.step(loss.detach())
        l = loss.item()
        if l < best_loss - 1e-4:
            best_loss = l
            stall = 0
        else:
            stall += 1
            if stall >= stagnation_patience:
                return False   # stuck — give up early
        with torch.no_grad():
            if torch.equal(out.argmax(dim=1), Y):
                return True
        # Early accuracy check: if after 100 epochs accuracy is < 5%, give up
        if epoch == 100:
            with torch.no_grad():
                acc = (out.argmax(dim=1) == Y).float().mean().item()
            if acc < 0.05:
                return False
    return False


def _is_exact_torch(model, X, Y, torch):
    model.eval()
    with torch.no_grad():
        return torch.equal(model(X).argmax(dim=1), Y)


def _export_torch(model, path, torch, onnx):
    model.eval()
    dummy = torch.zeros(1, C, CANVAS, CANVAS)
    try:
        torch.onnx.export(model.net, dummy, str(path),
            input_names=["input"], output_names=["output"],
            opset_version=17, dynamic_axes=None)
        m = onnx.load(str(path))
        onnx.checker.check_model(m)
        onnx_save(m, str(path))
        return True
    except Exception as e:
        print(f"      export error: {e}")
        return False


# ── Fast NumPy conv net ────────────────────────────────────────────────────────

def _fast_conv2d_fwd(x, W, b, pad):
    """
    Fast batched conv2d using einsum over kernel positions.
    x: [N, Cin, H, W], W: [Cout, Cin, kH, kW], b: [Cout]
    Returns out [N, Cout, H, W]
    """
    N, Cin, H, Ww = x.shape
    Cout, _, kH, kW = W.shape
    xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    out = np.zeros((N, Cout, H, Ww), dtype=np.float32)
    for i in range(kH):
        for j in range(kW):
            # xs: [N, Cin, H, W], wk: [Cout, Cin]
            xs = xp[:, :, i:i+H, j:j+Ww]
            wk = W[:, :, i, j]
            # 'nchw,oc->nohw'
            out += np.einsum('nchw,oc->nohw', xs, wk, optimize=True)
    out += b[None, :, None, None]
    return out


def _fast_conv2d_bwd(dout, x, W, pad):
    """
    Backward pass for fast_conv2d.
    Returns (dx, dW, db).
    """
    N, Cin, H, Ww = x.shape
    Cout, _, kH, kW = W.shape
    xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    dxp = np.zeros_like(xp)
    dW  = np.zeros_like(W)
    db  = dout.sum(axis=(0,2,3))

    for i in range(kH):
        for j in range(kW):
            xs  = xp[:, :, i:i+H, j:j+Ww]
            wk  = W[:, :, i, j]
            # dW[:,:,i,j] = einsum('nohw,nchw->oc', dout, xs)
            dW[:, :, i, j] = np.einsum('nohw,nchw->oc', dout, xs, optimize=True)
            # dxp[:,:,i:i+H,j:j+W] += einsum('nohw,oc->nchw', dout, wk)
            dxp[:, :, i:i+H, j:j+Ww] += np.einsum('nohw,oc->nchw', dout, wk, optimize=True)

    dx = dxp[:, :, pad:pad+H, pad:pad+Ww]
    return dx, dW, db


class NumpyConvNet:
    """Fast pure-numpy conv net using einsum-based forward/backward."""

    def __init__(self, hidden, kernel, depth, seed=42):
        np.random.seed(seed)
        self.kernel = kernel
        self.pad    = kernel // 2
        self.depth  = depth
        self.layers = []    # list of [W, b]
        self.use_relu = []

        in_c = C
        for _ in range(depth - 1):
            s = np.sqrt(2.0 / (in_c * kernel * kernel))
            W = (np.random.randn(hidden, in_c, kernel, kernel) * s).astype(np.float32)
            b = np.zeros(hidden, dtype=np.float32)
            self.layers.append([W, b])
            self.use_relu.append(True)
            in_c = hidden
        # output layer
        s = np.sqrt(2.0 / (in_c * kernel * kernel))
        W = (np.random.randn(C, in_c, kernel, kernel) * s).astype(np.float32)
        b = np.zeros(C, dtype=np.float32)
        self.layers.append([W, b])
        self.use_relu.append(False)

        # Adam state
        self._adam_m = [[np.zeros_like(l[0]), np.zeros_like(l[1])] for l in self.layers]
        self._adam_v = [[np.zeros_like(l[0]), np.zeros_like(l[1])] for l in self.layers]
        self._adam_t = 0

    def forward(self, x):
        self._cache = []
        cur = x
        for i, ((W, b), use_r) in enumerate(zip(self.layers, self.use_relu)):
            out = _fast_conv2d_fwd(cur, W, b, self.pad)
            pre = out.copy()
            if use_r:
                out = np.maximum(0.0, out)
            self._cache.append((cur, pre))
            cur = out
        return cur

    def backward(self, dout):
        grads = []
        for i in range(len(self.layers) - 1, -1, -1):
            x_in, pre_act = self._cache[i]
            if self.use_relu[i]:
                dout = dout * (pre_act > 0)
            W, _ = self.layers[i]
            dx, dW, db = _fast_conv2d_bwd(dout, x_in, W, self.pad)
            grads.insert(0, (dW, db))
            dout = dx
        return grads

    def step_adam(self, grads, lr):
        self._adam_t += 1
        t = self._adam_t
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (dW, db) in enumerate(grads):
            for j, d in enumerate([dW, db]):
                self._adam_m[i][j] = b1 * self._adam_m[i][j] + (1-b1) * d
                self._adam_v[i][j] = b2 * self._adam_v[i][j] + (1-b2) * d**2
                mh = self._adam_m[i][j] / (1 - b1**t)
                vh = self._adam_v[i][j] / (1 - b2**t)
                self.layers[i][j] -= lr * mh / (np.sqrt(vh) + eps)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def to_onnx(self, path):
        from onnx import numpy_helper, TensorProto, helper as oh
        nodes, inits = [], []
        cur = "input"
        for i, ((W, b), use_r) in enumerate(zip(self.layers, self.use_relu)):
            w_n, b_n = f"l{i}_W", f"l{i}_b"
            o_n = f"l{i}_conv"
            inits += [numpy_helper.from_array(W.astype(np.float32), w_n),
                      numpy_helper.from_array(b.astype(np.float32), b_n)]
            nodes.append(oh.make_node("Conv",
                inputs=[cur, w_n, b_n], outputs=[o_n],
                kernel_shape=[self.kernel, self.kernel],
                pads=[self.pad]*4))
            if use_r:
                r_n = f"l{i}_relu"
                nodes.append(oh.make_node("Relu", inputs=[o_n], outputs=[r_n]))
                cur = r_n
            else:
                cur = o_n
        graph = oh.make_graph(nodes, "learned",
            [oh.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
            [oh.make_tensor_value_info(cur,      TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
            inits)
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("",17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx_save(model, str(path))
        return True


# ── Dataset builder ────────────────────────────────────────────────────────────

def make_batch_np(pairs):
    xs, ys = [], []
    for p in pairs:
        xs.append(grid_to_tensor(p["input"]))
        out = grid_to_array(p["output"])
        H, W = out.shape
        tgt = np.zeros((CANVAS, CANVAS), dtype=np.int64)
        tgt[:H, :W] = out
        ys.append(tgt)
    X = np.concatenate(xs, axis=0)       # [N,10,30,30]
    Y = np.stack(ys, axis=0)             # [N,30,30]
    return X, Y


def _cross_entropy(logits, targets):
    """logits [N,C,H,W], targets [N,H,W] → scalar loss, dlogits"""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    N, Cc, H, W = logits.shape
    oh = np.zeros_like(probs)
    oh[np.arange(N)[:, None, None], targets,
       np.arange(H)[None, :, None], np.arange(W)[None, None, :]] = 1.0
    loss = -np.mean(np.log(probs.clip(1e-9)) * oh)
    grad = (probs - oh) / (N * H * W)
    return loss, grad


# ── Architecture search ────────────────────────────────────────────────────────

ARCH_TRIALS = [
    {"hidden": C,  "kernel": 1, "depth": 1, "restarts": 1, "stage1_epochs": 400, "stage2_epochs": 800},
    {"hidden": C,  "kernel": 3, "depth": 1, "restarts": 2, "stage1_epochs": 400, "stage2_epochs": 800},
    {"hidden": 16, "kernel": 3, "depth": 2, "restarts": 2, "stage1_epochs": 400, "stage2_epochs": 800},
    {"hidden": C,  "kernel": 5, "depth": 1, "restarts": 1, "stage1_epochs": 400, "stage2_epochs": 800},
    {"hidden": 32, "kernel": 3, "depth": 2, "restarts": 2, "stage1_epochs": 450, "stage2_epochs": 900},
    {"hidden": 16, "kernel": 5, "depth": 2, "restarts": 1, "stage1_epochs": 450, "stage2_epochs": 900},
    {"hidden": C,  "kernel": 7, "depth": 1, "restarts": 1, "stage1_epochs": 450, "stage2_epochs": 900},
    {"hidden": 16, "kernel": 3, "depth": 3, "restarts": 1, "stage1_epochs": 500, "stage2_epochs": 1000},
    {"hidden": 32, "kernel": 3, "depth": 3, "restarts": 1, "stage1_epochs": 500, "stage2_epochs": 1000},
    {"hidden": 32, "kernel": 5, "depth": 2, "restarts": 1, "stage1_epochs": 550, "stage2_epochs": 1100},
    {"hidden": 16, "kernel": 7, "depth": 2, "restarts": 1, "stage1_epochs": 550, "stage2_epochs": 1100},
    {"hidden": 16, "kernel": 5, "depth": 3, "restarts": 1, "stage1_epochs": 600, "stage2_epochs": 1200},
    {"hidden": 32, "kernel": 5, "depth": 3, "restarts": 1, "stage1_epochs": 650, "stage2_epochs": 1300},
    {"hidden": 64, "kernel": 3, "depth": 3, "restarts": 1, "stage1_epochs": 650, "stage2_epochs": 1300},
    {"hidden": C,  "kernel": 9, "depth": 1, "restarts": 1, "stage1_epochs": 550, "stage2_epochs": 1100},
    {"hidden": 32, "kernel": 3, "depth": 4, "restarts": 1, "stage1_epochs": 700, "stage2_epochs": 1400, "use_coords": True},
    {"hidden": 64, "kernel": 3, "depth": 4, "restarts": 1, "stage1_epochs": 750, "stage2_epochs": 1500, "use_coords": True},
    {"hidden": 32, "kernel": 5, "depth": 4, "restarts": 1, "stage1_epochs": 750, "stage2_epochs": 1500, "use_coords": True},
    {"hidden": 64, "kernel": 3, "depth": 4, "restarts": 1, "stage1_epochs": 800, "stage2_epochs": 1600, "use_coords": True, "residual": True},
]


# ── Training ───────────────────────────────────────────────────────────────────

def _train_numpy(model, X, Y, max_epochs=600, lr=3e-3, stagnation=120, deadline=None):
    """Train NumpyConvNet. Returns True when pixel-perfect."""
    import time
    best_loss = np.inf
    stall = 0
    cur_lr = lr
    lr_patience = 80
    lr_stall = 0

    for epoch in range(max_epochs):
        if deadline is not None and time.time() > deadline:
            return False
        logits = model.forward(X)
        if epoch == 100:
            acc = (np.argmax(logits, axis=1) == Y).mean()
            if acc < 0.05:
                return False
        loss, dlogits = _cross_entropy(logits, Y)
        grads = model.backward(dlogits)
        model.step_adam(grads, cur_lr)

        if loss < best_loss - 1e-4:
            best_loss = loss
            stall = 0
            lr_stall = 0
        else:
            stall += 1
            lr_stall += 1
            if stall >= stagnation:
                return False
            if lr_stall >= lr_patience:
                cur_lr = max(cur_lr * 0.5, 1e-5)
                lr_stall = 0

        if np.array_equal(np.argmax(logits, axis=1), Y):
            return True

    return False


# ── Solver ─────────────────────────────────────────────────────────────────────

class LearnedSolver(BaseSolver):
    PRIORITY = 90   # last resort

    def can_solve(self, _):
        return True

    def build(self, task_id, task, analysis, out_dir):
        train_pairs  = task.get("train",   [])
        test_pairs   = task.get("test",    [])
        arcgen_pairs = task.get("arc-gen", [])
        all_pairs    = train_pairs + test_pairs + arcgen_pairs

        if not train_pairs:
            return None

        # Skip tasks with grids larger than the 30×30 canvas
        for p in all_pairs:
            H = len(p["input"]); W = len(p["input"][0])
            if H > CANVAS or W > CANVAS:
                return None
            oH = len(p["output"]); oW = len(p["output"][0])
            if oH > CANVAS or oW > CANVAS:
                return None

        torch_mod, nn_mod, optim_mod = _try_import_torch()
        if torch_mod is not None:
            return self._build_torch(task_id, all_pairs, train_pairs, out_dir,
                                     torch_mod, nn_mod, optim_mod)
        else:
            return self._build_numpy(task_id, all_pairs, train_pairs, out_dir)

    # ── PyTorch path ────────────────────────────────────────────────────────

    def _build_torch(self, task_id, all_pairs, train_pairs, out_dir, torch, nn, optim):
        import time
        X_tr = torch.from_numpy(make_batch_np(train_pairs)[0])
        Y_tr = torch.from_numpy(make_batch_np(train_pairs)[1])
        X_all_np, Y_all_np = make_batch_np(all_pairs)
        X_all = torch.from_numpy(X_all_np)
        Y_all = torch.from_numpy(Y_all_np)
        path = out_dir / f"{task_id}.onnx"

        # Per-task deadline: 300 seconds total
        task_deadline = time.time() + 300.0

        for trial in ARCH_TRIALS:
            if time.time() > task_deadline:
                break
            hidden = trial["hidden"]
            kernel = trial["kernel"]
            depth = trial["depth"]
            use_coords = bool(trial.get("use_coords"))
            residual = bool(trial.get("residual"))
            for seed in range(trial["restarts"]):
                if time.time() > task_deadline:
                    break
                torch.manual_seed(seed)
                label = f"h={hidden} k={kernel} d={depth} s={seed}" + (" coord" if use_coords else "") + (" res" if residual else "")

                # Stage 1: train-only
                model = _CoordResNet(hidden, kernel, depth, torch, nn, use_coords=use_coords, residual=residual)
                if not _train_torch(
                    model, X_tr, Y_tr, trial["stage1_epochs"], 3e-3, torch, optim, nn,
                    deadline=task_deadline
                ):
                    continue
                if _is_exact_torch(model, X_all, Y_all, torch):
                    if _export_torch(model, path, torch, onnx):
                        print(f"      ✓ {label}  (stage-1 generalised)")
                        return path

                if time.time() > task_deadline:
                    break

                # Stage 1.5: continue from the train-fit model on all known pairs.
                if _train_torch(
                    model, X_all, Y_all, trial["stage2_epochs"], 1e-3, torch, optim, nn,
                    deadline=task_deadline
                ):
                    if _export_torch(model, path, torch, onnx):
                        print(f"      ✓ {label}  (stage-1.5 finetuned all-pairs)")
                        return path

                if time.time() > task_deadline:
                    break

                # Stage 2: fresh all-pairs fit
                torch.manual_seed(seed + 1000)
                model2 = _CoordResNet(hidden, kernel, depth, torch, nn, use_coords=use_coords, residual=residual)
                if _train_torch(
                    model2, X_all, Y_all, trial["stage2_epochs"], 1e-3, torch, optim, nn,
                    deadline=task_deadline
                ):
                    if _export_torch(model2, path, torch, onnx):
                        print(f"      ✓ {label}  (stage-2 all-pairs)")
                        return path
        return None

    # ── NumPy path ──────────────────────────────────────────────────────────

    def _build_numpy(self, task_id, all_pairs, train_pairs, out_dir):
        import time
        X_tr, Y_tr = make_batch_np(train_pairs)
        X_all, Y_all = make_batch_np(all_pairs)
        path = out_dir / f"{task_id}.onnx"
        task_deadline = time.time() + 180.0

        for trial in ARCH_TRIALS:
            if time.time() > task_deadline:
                break
            hidden = trial["hidden"]
            kernel = trial["kernel"]
            depth = trial["depth"]
            label = f"h={hidden} k={kernel} d={depth}"

            for seed in range(trial["restarts"]):
                if time.time() > task_deadline:
                    break

                # Stage 1: train-only (fast, few pairs)
                model = NumpyConvNet(hidden, kernel, depth, seed=42 + seed)
                if not _train_numpy(
                    model, X_tr, Y_tr, max_epochs=trial["stage1_epochs"],
                    lr=3e-3, deadline=task_deadline
                ):
                    continue
                if np.array_equal(model.predict(X_all), Y_all):
                    try:
                        model.to_onnx(path)
                        print(f"      ✓ {label} s={seed}  (stage-1 generalised, numpy)")
                        return path
                    except Exception as e:
                        print(f"      export error: {e}")

                if time.time() > task_deadline:
                    break

                # Stage 2: all-pairs (slower)
                model2 = NumpyConvNet(hidden, kernel, depth, seed=4200 + seed)
                if _train_numpy(
                    model2, X_all, Y_all, max_epochs=trial["stage2_epochs"],
                    lr=1e-3, deadline=task_deadline
                ):
                    try:
                        model2.to_onnx(path)
                        print(f"      ✓ {label} s={seed}  (stage-2 all-pairs, numpy)")
                        return path
                    except Exception as e:
                        print(f"      export error: {e}")

        return None
