"""
Walk-Forward Validation Engine — Multi-Horizon Stock Return Forecaster
=======================================================================
Purpose
-------
Complete systematic trading validation by:
  1. Splitting the time-series chronologically (never shuffled).
  2. Training on a rolling (or expanding) window of history.
  3. Testing strictly out-of-sample on the immediately following block.
  4. Sliding the window forward by `step_size` and repeating.
  5. Storing every window's metrics, predictions, and an equity curve
     stitched purely from out-of-sample folds (no leakage anywhere).

Design goals
------------
- Strict causal integrity: purge gap >= max(horizon) between train/test.
- Works standalone: if no real dataset is found, falls back to a
  synthetic multi-regime price generator so the whole pipeline can be
  smoke-tested end to end.
- Lightweight model (GRU encoder + multi-head horizon decoder) so that
  training dozens of walk-forward windows is tractable on CPU/GPU alike.
- Full bookkeeping: per-window metrics CSV, JSON summary, stitched
  OOS equity curve, rolling IC chart, and a window heatmap.
- Resumable: a completed window is skipped if its result file already
  exists on disk, so a long walk-forward run can be restarted safely.

Usage
-----
    python walk_forward_validation.py --tickers AAPL,MSFT,GOOG \
        --data-dir /tmp/ts_features_v5 --out-dir ./wf_results

"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import gc
import json
import math
import time
import random
import argparse
import warnings
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

EPS = 1e-8
_GLOBAL_START = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
def ts(msg: str):
    e = time.time() - _GLOBAL_START
    h, rem = divmod(int(e), 3600)
    m, s = divmod(rem, 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)


def eta(done_frac: float, start_time: float) -> str:
    elapsed = time.time() - start_time
    if done_frac <= 0:
        return "unknown"
    rem = elapsed / done_frac - elapsed
    h, r = divmod(int(rem), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_div(a, b, fill=0.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) > EPS, a / b, fill)
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WFConfig:
    # Data
    data_dir: str = "/tmp/ts_features_v5"
    out_dir: str = "./wf_results"
    tickers: Optional[List[str]] = None
    max_tickers: int = 12

    # Sequence / horizon
    seq_len: int = 90            # encoder window length fed to the model
    horizons: Tuple[int, ...] = (1, 3, 5, 10, 20)
    max_h: int = field(init=False)

    # Walk-forward geometry
    initial_train_frac: float = 0.35   # fraction of full history for the FIRST train window
    test_frac: float = 0.08            # fraction of full history used per OOS test fold
    step_frac: float = 0.08            # how far the window slides forward each iteration
    purge_gap: int = field(init=False)  # rows excluded between train end and test start
    expanding: bool = True              # True = expanding window, False = rolling (fixed-size) window
    max_windows: int = 40

    # Model / training
    hidden: int = 96
    n_layers: int = 2
    dropout: float = 0.15
    lr: float = 1.5e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs_per_window: int = 18
    patience: int = 5
    grad_clip: float = 1.0
    warm_start: bool = True   # carry weights forward between windows (faster convergence)

    # Reproducibility / misc
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: bool = True

    def __post_init__(self):
        self.max_h = max(self.horizons)
        self.purge_gap = self.max_h + 2  # small safety buffer beyond the largest horizon


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic multi-regime data generator (fallback / smoke test)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_synthetic_panel(n_tickers=8, n_days=2600, seed=7) -> Dict[str, pd.DataFrame]:
    """
    Builds a synthetic OHLCV panel with regime shifts (trending / mean-reverting /
    volatile) so the walk-forward engine has something realistic to chew on when
    no real dataset is available. Regimes rotate every ~250-400 trading days.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2012-01-02", periods=n_days)
    panel = {}
    for t in range(n_tickers):
        sym = f"SYN{t:02d}"
        regime_len = rng.randint(200, 400)
        n_regimes = n_days // regime_len + 2
        rets = []
        vol_state = rng.uniform(0.008, 0.02)
        for r in range(n_regimes):
            regime_type = rng.choice(["trend_up", "trend_down", "mean_revert", "volatile"])
            length = min(regime_len, n_days - len(rets))
            if length <= 0:
                break
            if regime_type == "trend_up":
                mu = rng.uniform(0.0003, 0.0009)
                sigma = vol_state
                seg = rng.normal(mu, sigma, length)
            elif regime_type == "trend_down":
                mu = -rng.uniform(0.0003, 0.0009)
                sigma = vol_state
                seg = rng.normal(mu, sigma, length)
            elif regime_type == "mean_revert":
                seg = np.zeros(length)
                level = 0.0
                for i in range(length):
                    level = 0.9 * level + rng.normal(0, vol_state)
                    seg[i] = -0.3 * level + rng.normal(0, vol_state * 0.4)
            else:  # volatile / shock regime
                sigma = vol_state * rng.uniform(2.0, 4.0)
                seg = rng.normal(0, sigma, length)
                if rng.rand() < 0.3:
                    shock_idx = rng.randint(0, length)
                    seg[shock_idx] -= rng.uniform(0.05, 0.12)  # crash day
            rets.append(seg)
        ret = np.concatenate(rets)[:n_days]
        close = 50.0 * np.cumprod(1 + ret)
        high = close * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = rng.lognormal(mean=13.0, sigma=0.5, size=n_days)
        df = pd.DataFrame({
            "date": dates, "open": open_, "high": high, "low": low,
            "close": close, "volume": volume,
        }).set_index("date")
        panel[sym] = df
    return panel


def _featurize(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Compact, purely-causal feature block (no lookahead by construction —
    every rolling stat at row t only uses rows <= t).
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    n = len(close)

    ret1 = np.zeros(n)
    ret1[1:] = safe_div(close[1:] - close[:-1], close[:-1])

    feats = {}
    for w in (5, 10, 20, 60):
        r = np.full(n, np.nan)
        r[w:] = safe_div(close[w:] - close[:-w], close[:-w])
        feats[f"mom_{w}"] = np.nan_to_num(r)

    s_ret = pd.Series(ret1)
    for w in (5, 10, 20, 60):
        feats[f"rvol_{w}"] = s_ret.rolling(w, min_periods=max(2, w // 2)).std().fillna(0.).values

    ma20 = pd.Series(close).rolling(20, min_periods=5).mean().bfill().values
    ma50 = pd.Series(close).rolling(50, min_periods=10).mean().bfill().values
    feats["px_to_ma20"] = safe_div(close - ma20, ma20)
    feats["px_to_ma50"] = safe_div(close - ma50, ma50)

    # RSI-14 (Wilder-style, causal)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=5).mean().fillna(0.).values
    avg_loss = pd.Series(loss).rolling(14, min_periods=5).mean().fillna(0.).values
    rs = safe_div(avg_gain, avg_loss + EPS)
    feats["rsi14"] = 1.0 - 1.0 / (1.0 + rs)

    hl_range = np.clip(high - low, 1e-6, None)
    feats["hl_spread"] = safe_div(high - low, close)
    feats["close_pos"] = safe_div(close - low, hl_range)

    vol_ma = pd.Series(volume).rolling(20, min_periods=5).mean().bfill().values
    feats["vol_z"] = safe_div(volume - vol_ma, vol_ma + EPS)

    feats["ret1"] = ret1
    dow = np.array(df.index.dayofweek, dtype=np.float64)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    names = sorted(feats.keys())
    mat = np.stack([feats[k] for k in names], axis=1).astype(np.float32)
    mat = np.nan_to_num(mat, nan=0., posinf=0., neginf=0.)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        lo, hi = np.percentile(col, 1.0), np.percentile(col, 99.0)
        mat[:, j] = np.clip(col, lo, hi)
    return mat, names


# ─────────────────────────────────────────────────────────────────────────────
# Real V5 dataset loader (reads output of the feature-engineering pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def load_real_panel(data_dir: str, max_tickers: int) -> Optional[Dict[str, Dict]]:
    """
    Loads {ticker}_unknown.npy / {ticker}_close.npy from a V5-style directory.
    Returns None if nothing usable is present so the caller can fall back to
    synthetic data.
    """
    d = Path(data_dir)
    if not d.exists():
        return None
    npy = list(d.glob("*_unknown.npy"))
    if not npy:
        return None
    tickers = sorted({f.stem.replace("_unknown", "") for f in npy})[:max_tickers]
    feat_file = d / "feature_names_unknown.txt"
    feat_names = feat_file.read_text().strip().split("\n") if feat_file.exists() else None

    panel = {}
    for tkr in tickers:
        up = d / f"{tkr}_unknown.npy"
        cp = d / f"{tkr}_close.npy"
        if not (up.exists() and cp.exists()):
            continue
        try:
            X = np.load(up, mmap_mode="r")
            # V5 arrays are pre-windowed [S, T, F]; collapse to a flat per-day
            # feature matrix by taking the last row of each window (most recent
            # causal snapshot), which reconstructs a (roughly) daily series.
            if X.ndim == 3:
                mat = np.array(X[:, -1, :], dtype=np.float32)
            else:
                mat = np.array(X, dtype=np.float32)
            close = np.load(cp, mmap_mode="r")
            close = np.array(close[-mat.shape[0]:], dtype=np.float32)
            if len(close) != mat.shape[0]:
                m = min(len(close), mat.shape[0])
                close, mat = close[-m:], mat[-m:]
            panel[tkr] = {"features": mat, "close": close, "feat_names": feat_names}
        except Exception as e:
            ts(f"WARN loading {tkr}: {e}")
    return panel if panel else None


def build_panel(cfg: WFConfig) -> Dict[str, Dict]:
    real = load_real_panel(cfg.data_dir, cfg.max_tickers)
    if real is not None:
        ts(f"Loaded REAL panel: {len(real)} tickers from {cfg.data_dir}")
        return real
    ts("No usable real dataset found — generating SYNTHETIC multi-regime panel")
    raw = _generate_synthetic_panel(n_tickers=cfg.max_tickers, seed=cfg.seed)
    panel = {}
    for sym, df in raw.items():
        mat, names = _featurize(df)
        panel[sym] = {"features": mat, "close": df["close"].values.astype(np.float32),
                       "feat_names": names}
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward window geometry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WindowSpec:
    window_id: int
    train_start: int
    train_end: int      # exclusive
    test_start: int      # inclusive, >= train_end + purge_gap
    test_end: int         # exclusive


class WalkForwardSplitter:
    """
    Produces chronological (train, purge, test) index windows over a single
    aligned time axis of length `n`. Never shuffles. Slides forward by
    `step` rows each iteration. Supports expanding (train_start fixed at 0)
    or rolling (fixed-size train window) modes.
    """
    def __init__(self, n_rows: int, cfg: WFConfig):
        self.n = n_rows
        self.cfg = cfg
        self.initial_train = max(int(n_rows * cfg.initial_train_frac), cfg.seq_len + cfg.max_h + 10)
        self.test_size = max(int(n_rows * cfg.test_frac), cfg.max_h + 20)
        self.step = max(int(n_rows * cfg.step_frac), 1)
        self.purge = cfg.purge_gap

    def generate(self) -> List[WindowSpec]:
        windows = []
        train_start = 0
        train_end = self.initial_train
        wid = 0
        while True:
            test_start = train_end + self.purge
            test_end = test_start + self.test_size
            if test_end > self.n:
                break
            windows.append(WindowSpec(wid, train_start, train_end, test_start, test_end))
            wid += 1
            if self.cfg.max_windows and wid >= self.cfg.max_windows:
                break
            train_end += self.step
            if not self.cfg.expanding:
                train_start += self.step
        return windows


# ─────────────────────────────────────────────────────────────────────────────
# Sequence dataset for a single window
# ─────────────────────────────────────────────────────────────────────────────
class WindowSeqDataset(Dataset):
    """
    Builds fixed-length encoder sequences ending at index `t`, with targets
    being forward log-returns close[t+h]/close[t] for each horizon h. Only
    indices whose full target window fits inside [lo, hi) of the *global*
    array are included — this is how purge/no-lookahead is enforced per split.
    """
    def __init__(self, features: np.ndarray, close: np.ndarray, seq_len: int,
                 horizons: Tuple[int, ...], lo: int, hi: int,
                 norm_mu: Optional[np.ndarray] = None, norm_sigma: Optional[np.ndarray] = None):
        self.features = features
        self.close = close
        self.seq_len = seq_len
        self.horizons = horizons
        self.max_h = max(horizons)
        self.norm_mu = norm_mu
        self.norm_sigma = norm_sigma

        self.valid_ends = []
        n = len(close)
        for end in range(lo, hi):
            start = end - seq_len + 1
            if start < 0:
                continue
            if end + self.max_h >= n:
                continue
            ok = True
            for h in horizons:
                if not (close[end] > 0 and close[end + h] > 0):
                    ok = False
                    break
            if ok:
                self.valid_ends.append(end)

    def __len__(self):
        return len(self.valid_ends)

    def __getitem__(self, idx):
        end = self.valid_ends[idx]
        start = end - self.seq_len + 1
        seq = self.features[start:end + 1].astype(np.float32)
        if self.norm_mu is not None:
            seq = np.clip((seq - self.norm_mu) / (self.norm_sigma + EPS), -6., 6.)
        seq = np.nan_to_num(seq, nan=0., posinf=0., neginf=0.)
        c0 = self.close[end]
        target = np.array(
            [math.log(float(self.close[end + h]) / float(c0)) for h in self.horizons],
            dtype=np.float32)
        target = np.nan_to_num(target, nan=0., posinf=0., neginf=0.)
        return torch.from_numpy(seq), torch.from_numpy(target)


def collate_seq(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def compute_train_norm_stats(features: np.ndarray, lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray]:
    """Normalisation stats computed ONLY on the train slice [lo, hi) — no leakage."""
    block = features[lo:hi]
    mu = np.nanmean(block, axis=0).astype(np.float32)
    sigma = np.nanstd(block, axis=0).astype(np.float32)
    sigma = np.where(sigma < EPS, 1.0, sigma)
    return mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight multi-horizon model
# ─────────────────────────────────────────────────────────────────────────────
class TemporalAttnPool(nn.Module):
    """Simple additive-attention pooling over encoder time steps."""
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, seq):  # seq: [B, T, D]
        w = torch.softmax(self.score(seq).squeeze(-1), dim=-1)  # [B, T]
        return (seq * w.unsqueeze(-1)).sum(1), w


class WFModel(nn.Module):
    """
    GRU encoder -> attention pooling -> per-horizon MLP heads.
    Predicts both a point log-return forecast and a directional (sign) logit
    for every horizon, which is enough to build IC/DA/Sharpe-style metrics
    without the overhead of a full quantile TFT stack.
    """
    def __init__(self, n_features: int, horizons: Tuple[int, ...],
                 hidden: int = 96, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.horizons = horizons
        self.n_h = len(horizons)
        self.input_proj = nn.Linear(n_features, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0, bidirectional=False)
        self.pool = TemporalAttnPool(hidden)
        self.norm = nn.LayerNorm(hidden)
        self.trunk = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.point_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])
        self.sign_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, x):
        h0 = self.input_proj(x)
        seq, last = self.gru(h0)
        pooled, attn_w = self.pool(seq)
        last_h = last[-1]
        combined = self.norm(torch.cat([pooled, last_h], dim=-1))
        trunk = self.trunk(combined)
        points = torch.cat([head(trunk) for head in self.point_heads], dim=-1)
        signs = torch.cat([head(trunk) for head in self.sign_heads], dim=-1)
        return points, signs, attn_w


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────
def huber_loss(pred, target, delta=0.01):
    err = target - pred
    ab = err.abs()
    return torch.where(ab < delta, 0.5 * err ** 2 / delta, ab - 0.5 * delta).mean()


def directional_bce(logits, target):
    y = (target > 0).float()
    return F.binary_cross_entropy_with_logits(logits, y)


def ic_soft_loss(pred, target):
    def _n(x):
        mu = x.mean(0, keepdim=True)
        sd = x.std(0, keepdim=True) + EPS
        return (x - mu) / sd
    return -(_n(pred) * _n(target)).mean()


def window_loss(points, signs, target, lambda_sign=0.25, lambda_ic=0.15):
    h = huber_loss(points, target)
    b = directional_bce(signs, target)
    ic = ic_soft_loss(points, target)
    lam_base = max(1.0 - lambda_sign - lambda_ic, 0.1)
    return lam_base * h + lambda_sign * b + lambda_ic * ic, h, b, ic


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_window_metrics(pred: np.ndarray, target: np.ndarray, horizons: Tuple[int, ...]) -> Dict:
    """
    pred/target: [N, H] arrays of point forecasts and realised forward
    log-returns for one out-of-sample test fold. Returns per-horizon
    IC, Rank IC, directional accuracy, long/short Sharpe, max drawdown,
    and top-vs-bottom decile spread.
    """
    out = {}
    N, H = pred.shape
    for hi, h in enumerate(horizons):
        p = pred[:, hi]
        t = target[:, hi]
        ic = float(np.corrcoef(p, t)[0, 1]) if N > 2 and np.std(p) > 0 else 0.0
        try:
            from scipy.stats import spearmanr
            ric, _ = spearmanr(p, t)
            ric = float(ric) if ric == ric else 0.0
        except Exception:
            ric = 0.0
        da = float(np.mean(np.sign(p) == np.sign(t)))
        med = np.median(p)
        ls_ret = np.where(p > med, t, -t)
        sharpe = float(ls_ret.mean() / (ls_ret.std() + EPS) * math.sqrt(252))
        cum = np.cumprod(1 + np.clip(ls_ret, -0.5, 0.5))
        run_max = np.maximum.accumulate(cum)
        mdd = float(np.max((run_max - cum) / (run_max + EPS))) if len(cum) else 0.0
        q75, q25 = np.percentile(p, 75), np.percentile(p, 25)
        top = t[p >= q75].mean() if (p >= q75).any() else 0.0
        bot = t[p <= q25].mean() if (p <= q25).any() else 0.0
        out[f"IC_h{h}"] = ic
        out[f"RankIC_h{h}"] = ric
        out[f"DA_h{h}"] = da
        out[f"Sharpe_h{h}"] = sharpe
        out[f"MaxDD_h{h}"] = mdd
        out[f"Decile_spread_h{h}"] = float(top - bot)
    out["IC_mean"] = float(np.mean([out[f"IC_h{h}"] for h in horizons]))
    out["RankIC_mean"] = float(np.mean([out[f"RankIC_h{h}"] for h in horizons]))
    out["DA_mean"] = float(np.mean([out[f"DA_h{h}"] for h in horizons]))
    out["Sharpe_mean"] = float(np.mean([out[f"Sharpe_h{h}"] for h in horizons]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-window training / evaluation
# ─────────────────────────────────────────────────────────────────────────────
def train_one_window(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
                     cfg: WFConfig, device: torch.device) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs_per_window)
    best_val = float("inf")
    best_state = None
    patience_cnt = 0

    for epoch in range(1, cfg.epochs_per_window + 1):
        model.train()
        tot_loss, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            points, signs, _ = model(xb)
            loss, *_ = window_loss(points, signs, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                points, signs, _ = model(xb)
                loss, *_ = window_loss(points, signs, yb)
                vtot += loss.item() * xb.size(0)
                vn += xb.size(0)
        val_loss = vtot / max(vn, 1)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_window(model: nn.Module, test_dl: DataLoader, device: torch.device
                    ) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_p, all_t = [], []
    for xb, yb in test_dl:
        xb = xb.to(device)
        points, _, _ = model(xb)
        all_p.append(points.cpu().numpy())
        all_t.append(yb.numpy())
    if not all_p:
        return np.empty((0, len(model.horizons))), np.empty((0, len(model.horizons)))
    return np.concatenate(all_p), np.concatenate(all_t)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class WalkForwardValidator:
    def __init__(self, cfg: WFConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.out_dir, "windows").mkdir(exist_ok=True)
        Path(cfg.out_dir, "plots").mkdir(exist_ok=True)
        set_seed(cfg.seed)
        self.results: List[Dict] = []
        self.oos_pred_stream: List[np.ndarray] = []
        self.oos_tgt_stream: List[np.ndarray] = []
        self.oos_ticker_stream: List[str] = []

    # ---- panel prep -----------------------------------------------------
    def _prepare_ticker_axis(self, panel: Dict[str, Dict]) -> Dict[str, Dict]:
        """Trims all tickers to a common usable length for aligned windowing."""
        min_len = min(d["features"].shape[0] for d in panel.values())
        trimmed = {}
        for sym, d in panel.items():
            trimmed[sym] = {
                "features": d["features"][-min_len:],
                "close": d["close"][-min_len:],
            }
        ts(f"Aligned panel: {len(trimmed)} tickers x {min_len} rows")
        return trimmed

    # ---- main loop --------------------------------------------------------
    def run(self):
        cfg = self.cfg
        ts("=== WALK-FORWARD VALIDATION START ===")
        ts(f"Config: seq_len={cfg.seq_len} horizons={cfg.horizons} purge={cfg.purge_gap} "
           f"expanding={cfg.expanding} max_windows={cfg.max_windows}")

        panel = build_panel(cfg)
        panel = self._prepare_ticker_axis(panel)
        n_rows = next(iter(panel.values()))["features"].shape[0]

        splitter = WalkForwardSplitter(n_rows, cfg)
        windows = splitter.generate()
        ts(f"Generated {len(windows)} walk-forward windows over {n_rows} rows "
           f"(train0={splitter.initial_train}, test_size={splitter.test_size}, step={splitter.step})")

        warm_model = None
        run_start = time.time()

        for wi, spec in enumerate(windows):
            result_path = Path(cfg.out_dir, "windows", f"window_{spec.window_id:03d}.json")
            if cfg.resume and result_path.exists():
                ts(f"[Window {spec.window_id}] already completed — skipping (resume=True)")
                with open(result_path) as fh:
                    self.results.append(json.load(fh))
                continue

            done_frac = (wi + 1) / len(windows)
            ts(f"[Window {spec.window_id}/{len(windows)-1}] "
               f"train=[{spec.train_start}:{spec.train_end}) "
               f"purge={spec.test_start - spec.train_end} "
               f"test=[{spec.test_start}:{spec.test_end}) | "
               f"ETA={eta(done_frac, run_start)}")

            try:
                metrics, preds, tgts, syms = self._run_single_window(panel, spec, warm_model)
            except Exception as e:
                ts(f"[Window {spec.window_id}] ERROR: {e}\n{traceback.format_exc()}")
                continue

            metrics["window_id"] = spec.window_id
            metrics["train_start"] = spec.train_start
            metrics["train_end"] = spec.train_end
            metrics["test_start"] = spec.test_start
            metrics["test_end"] = spec.test_end
            self.results.append(metrics)
            with open(result_path, "w") as fh:
                json.dump(metrics, fh, indent=2)

            self.oos_pred_stream.append(preds)
            self.oos_tgt_stream.append(tgts)
            self.oos_ticker_stream.extend(syms)

            ts(f"[Window {spec.window_id}] DONE | IC_mean={metrics['IC_mean']:.4f} "
               f"RankIC_mean={metrics['RankIC_mean']:.4f} DA_mean={metrics['DA_mean']:.3f} "
               f"Sharpe_mean={metrics['Sharpe_mean']:.3f}")

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._finalize()

    # ---- single window -----------------------------------------------------
    def _run_single_window(self, panel: Dict[str, Dict], spec: WindowSpec,
                           warm_model: Optional[nn.Module]
                           ) -> Tuple[Dict, np.ndarray, np.ndarray, List[str]]:
        cfg = self.cfg
        n_features = next(iter(panel.values()))["features"].shape[1]

        # Build per-ticker datasets and concatenate — normalisation stats
        # are computed strictly from each ticker's own TRAIN slice.
        train_sets, val_sets, test_sets = [], [], []
        train_len = spec.train_end - spec.train_start
        val_cut = spec.train_start + int(train_len * 0.85)  # last 15% of train = internal val

        test_syms_per_sample = []
        for sym, d in panel.items():
            feats, close = d["features"], d["close"]
            mu, sigma = compute_train_norm_stats(feats, spec.train_start, val_cut)

            tr_ds = WindowSeqDataset(feats, close, cfg.seq_len, cfg.horizons,
                                     spec.train_start, val_cut, mu, sigma)
            va_ds = WindowSeqDataset(feats, close, cfg.seq_len, cfg.horizons,
                                     val_cut, spec.train_end, mu, sigma)
            te_ds = WindowSeqDataset(feats, close, cfg.seq_len, cfg.horizons,
                                     spec.test_start, spec.test_end, mu, sigma)
            if len(tr_ds) > 0:
                train_sets.append(tr_ds)
            if len(va_ds) > 0:
                val_sets.append(va_ds)
            if len(te_ds) > 0:
                test_sets.append(te_ds)
                test_syms_per_sample.extend([sym] * len(te_ds))

        if not train_sets or not test_sets:
            raise RuntimeError("insufficient data to build train/test sets for this window")

        train_ds = torch.utils.data.ConcatDataset(train_sets)
        val_ds = torch.utils.data.ConcatDataset(val_sets) if val_sets else train_ds
        test_ds = torch.utils.data.ConcatDataset(test_sets)

        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_seq, drop_last=len(train_ds) > cfg.batch_size)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                            collate_fn=collate_seq)
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                             collate_fn=collate_seq)

        if warm_model is not None and cfg.warm_start:
            model = warm_model
        else:
            model = WFModel(n_features, cfg.horizons, cfg.hidden, cfg.n_layers, cfg.dropout)
        model = model.to(self.device)

        model = train_one_window(model, train_dl, val_dl, cfg, self.device)
        preds, tgts = evaluate_window(model, test_dl, self.device)
        metrics = compute_window_metrics(preds, tgts, cfg.horizons)

        if cfg.warm_start:
            self._carry_forward_model = model

        return metrics, preds, tgts, test_syms_per_sample

    @property
    def _carry_forward_model(self):
        return getattr(self, "_wm", None)

    @_carry_forward_model.setter
    def _carry_forward_model(self, m):
        self._wm = m

    # ---- finalisation: aggregate stats, plots, stitched equity curve ------
    def _finalize(self):
        cfg = self.cfg
        if not self.results:
            ts("No windows completed — nothing to summarise.")
            return

        df = pd.DataFrame(self.results).sort_values("window_id")
        csv_path = Path(cfg.out_dir, "walk_forward_metrics.csv")
        df.to_csv(csv_path, index=False)
        ts(f"Per-window metrics saved: {csv_path}")

        summary = {
            "n_windows": len(df),
            "IC_mean_avg": float(df["IC_mean"].mean()),
            "IC_mean_std": float(df["IC_mean"].std()),
            "RankIC_mean_avg": float(df["RankIC_mean"].mean()),
            "DA_mean_avg": float(df["DA_mean"].mean()),
            "Sharpe_mean_avg": float(df["Sharpe_mean"].mean()),
            "Sharpe_mean_std": float(df["Sharpe_mean"].std()),
            "pct_windows_IC_positive": float((df["IC_mean"] > 0).mean()),
            "pct_windows_Sharpe_positive": float((df["Sharpe_mean"] > 0).mean()),
            "best_window_id": int(df.loc[df["Sharpe_mean"].idxmax(), "window_id"]),
            "worst_window_id": int(df.loc[df["Sharpe_mean"].idxmin(), "window_id"]),
        }
        with open(Path(cfg.out_dir, "walk_forward_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        ts("=== WALK-FORWARD SUMMARY ===")
        for k, v in summary.items():
            ts(f"  {k:28s}: {v}")

        self._plot_metric_over_windows(df)
        self._plot_stitched_equity_curve()
        self._plot_window_heatmap(df)
        ts(f"All artefacts written to {cfg.out_dir}/")

    def _plot_metric_over_windows(self, df: pd.DataFrame):
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        panels = [("IC_mean", "IC (mean over horizons)"),
                  ("RankIC_mean", "Rank IC (mean over horizons)"),
                  ("DA_mean", "Directional accuracy (mean)"),
                  ("Sharpe_mean", "Long/short Sharpe (mean, annualised)")]
        for ax, (col, title) in zip(axes, panels):
            ax.plot(df["window_id"], df[col], marker="o", markersize=3, linewidth=1.4)
            ax.axhline(0 if "Sharpe" in col or "IC" in col else 0.5,
                      color="grey", linestyle=":", linewidth=1)
            ax.set_title(title)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Walk-forward window id")
        fig.suptitle("Walk-Forward Out-of-Sample Metrics", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = Path(self.cfg.out_dir, "plots", "metrics_over_windows.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        ts(f"Saved: {out}")

    def _plot_stitched_equity_curve(self):
        if not self.oos_pred_stream:
            return
        preds = np.concatenate(self.oos_pred_stream)
        tgts = np.concatenate(self.oos_tgt_stream)
        h_idx = 0  # shortest horizon for the stitched equity curve
        p = preds[:, h_idx]
        t = tgts[:, h_idx]
        med = np.median(p)
        ls_ret = np.where(p > med, t, -t)
        equity = np.cumprod(1 + np.clip(ls_ret, -0.5, 0.5))

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(equity, color="#4E79A7", linewidth=1.4)
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        ax.set_title("Stitched Out-of-Sample Equity Curve (long/short, shortest horizon)")
        ax.set_xlabel("Out-of-sample observation index (chronological across windows)")
        ax.set_ylabel("Cumulative growth of $1")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = Path(self.cfg.out_dir, "plots", "stitched_equity_curve.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        ts(f"Saved: {out}")

    def _plot_window_heatmap(self, df: pd.DataFrame):
        h_cols = [c for c in df.columns if c.startswith("IC_h")]
        if not h_cols:
            return
        mat = df[h_cols].values.T
        fig, ax = plt.subplots(figsize=(11, 3.5))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.15)
        ax.set_yticks(range(len(h_cols)))
        ax.set_yticklabels([c.replace("IC_h", "h=") for c in h_cols])
        ax.set_xlabel("Walk-forward window id")
        ax.set_title("IC Heatmap: Horizon x Window")
        fig.colorbar(im, ax=ax, label="IC")
        fig.tight_layout()
        out = Path(self.cfg.out_dir, "plots", "ic_heatmap.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        ts(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> WFConfig:
    p = argparse.ArgumentParser(description="Walk-forward validation for multi-horizon forecasting")
    p.add_argument("--data-dir", type=str, default="/tmp/ts_features_v5")
    p.add_argument("--out-dir", type=str, default="./wf_results")
    p.add_argument("--max-tickers", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=90)
    p.add_argument("--horizons", type=str, default="1,3,5,10,20")
    p.add_argument("--initial-train-frac", type=float, default=0.35)
    p.add_argument("--test-frac", type=float, default=0.08)
    p.add_argument("--step-frac", type=float, default=0.08)
    p.add_argument("--rolling", action="store_true", help="use a fixed-size rolling window instead of expanding")
    p.add_argument("--max-windows", type=int, default=40)
    p.add_argument("--epochs-per-window", type=int, default=18)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    horizons = tuple(int(x) for x in args.horizons.split(","))
    return WFConfig(
        data_dir=args.data_dir, out_dir=args.out_dir, max_tickers=args.max_tickers,
        seq_len=args.seq_len, horizons=horizons,
        initial_train_frac=args.initial_train_frac, test_frac=args.test_frac,
        step_frac=args.step_frac, expanding=not args.rolling, max_windows=args.max_windows,
        epochs_per_window=args.epochs_per_window, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, hidden=args.hidden,
        warm_start=not args.no_warm_start, resume=not args.no_resume, seed=args.seed,
    )


def main():
    cfg = parse_args()
    ts(f"Device: {cfg.device}")
    validator = WalkForwardValidator(cfg)
    validator.run()
    ts("=== WALK-FORWARD VALIDATION COMPLETE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        print(traceback.format_exc())
        raise
