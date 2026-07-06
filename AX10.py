
# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, gc, time, sys, math, json, random, warnings, shutil, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats as scipy_stats
from scipy.signal import welch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint as grad_ckpt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

warnings.filterwarnings("ignore")

# Set Plotly renderer — "notebook_connected" works in Kaggle/Colab
# Falls back silently to the default if not in a notebook.
try:
    import plotly.io as _pio_init
    _pio_init.renderers.default = "iframe"
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FLOAT16_MAX  = 65504.0
EPS          = 1e-8
WINSOR_PCT   = 0.5
SEQ_LEN      = 180          # total window written per sample
STEP         = 5            # stride between consecutive windows
OUT_DIR_V4   = "/tmp/ts_features_v4"
OUT_DIR_V5   = "/tmp/ts_features_v5"
BASE_INPUT_DIR = "/kaggle/input/datasets/adityaravojha/aurant-aseries"
RAM_LIMIT_GB   = 24.0       # conservative (was 27 — Kaggle sometimes reports RSS high)
STORAGE_LIMIT_GB = 48.0
VRAM_LIMIT_GB  = 14.5
HORIZONS       = [1, 3, 5, 7, 14, 28]
TARGET_N_FEATURES = 200
DARK           = "plotly_dark"
PURGE_GAP      = max(HORIZONS)   # rows excluded between train/val and val/test

DROP_RAW_COLUMNS = [
    "company","headquarters","date_added","founded",
    "sma_5","sma_10","sma_20","sma_50","sma_100","sma_200",
    "bb_mid","bb_upper","bb_lower","high20","low20",
    "up","down","trend_energy","ema_ratio","log_volume","log_dollar_volume",
    "cs_mean_ret","cs_std_ret","cs_dispersion","cs_mom_dispersion",
    "beta_change","market_excess",
]
CALENDAR_NAMES = [
    "cal_dow_sin","cal_dow_cos","cal_month_sin","cal_month_cos",
    "cal_woy_sin","cal_woy_cos","cal_quarter","cal_dom",
    "cal_month_end","cal_month_start",
]

_GLOBAL_START = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def ts(msg: str):
    e = time.time() - _GLOBAL_START
    h = int(e // 3600); m = int((e % 3600) // 60); s = int(e % 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)

def eta(done_frac: float, start_time=None) -> str:
    if start_time is None: start_time = _GLOBAL_START
    elapsed = time.time() - start_time
    if done_frac <= 0: return "unknown"
    rem = elapsed / done_frac - elapsed
    return f"{int(rem//3600):02d}h{int((rem%3600)//60):02d}m{int(rem%60):02d}s"

def _check_ram_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except: return 0.0

def _check_storage_gb(path="/tmp") -> float:
    try:
        total = 0.0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp): total += os.path.getsize(fp)
        return total / 1e9
    except: return 0.0

def _safe_delete(*paths):
    """Delete files silently, ignoring errors."""
    for p in paths:
        try:
            if os.path.exists(p): os.remove(p)
        except: pass

def _free_space_gb(path="/tmp") -> float:
    """Return free disk space in GB for the partition containing `path`."""
    try:
        import shutil as _sh
        total, used, free = _sh.disk_usage(path)
        return free / 1e9
    except: return 99.0   # assume OK if we can't check

def _check_free_before_write(path="/tmp", needed_gb=0.5) -> bool:
    """Return False (and log) if there is not enough free space to proceed."""
    free = _free_space_gb(path)
    if free < needed_gb:
        ts(f"DISK FULL: only {free:.2f}GB free on {path} — skipping write")
        return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Per-stock time-series chart (5 key stats)
# ─────────────────────────────────────────────────────────────────────────────
def save_stock_chart(sym: str, sdf: pd.DataFrame, mat: np.ndarray,
                     feat_names: list, out_dir: str):
    """
    For every processed stock:
      1. Print an ASCII stats summary to stdout (always visible in any environment)
      2. Call fig.show() so Kaggle/Jupyter renders it inline immediately
      3. Save the HTML to out_dir/stocks/{sym}_stats.html for later viewing

    5 panels:
      Close price + Kalman signal | Realised vol 20d vs 60d | RSI-14 |
      Rolling Sharpe-60 | VWAP deviation
    """
    try:
        stock_dir = os.path.join(out_dir, "stocks")
        os.makedirs(stock_dir, exist_ok=True)

        dates = list(sdf.index.astype(str))
        n     = len(dates)

        def _col(name):
            if name in feat_names:
                return mat[:, feat_names.index(name)]
            return np.zeros(n, dtype=np.float32)

        close_arr = sdf["close"].values.astype(np.float32)
        rv20      = _col("realized_vol_20")
        rv60      = _col("realized_vol_60")
        rsi14     = _col("rsi14")
        sharpe60  = _col("sharpe_60")
        vwap_dev  = _col("vwap_deviation")
        kf_dev    = _col("kalman_filtered_price")  # tanh deviation

        # ── 1. ASCII summary (always visible) ────────────────────────────────
        last_close  = float(close_arr[-1])
        ret_ytd     = float((close_arr[-1] / close_arr[0] - 1) * 100) if close_arr[0] > 0 else 0.
        cur_rv20    = float(rv20[-1]) * 100
        cur_rsi     = float(rsi14[-1])
        cur_sharpe  = float(sharpe60[-1]) * 5   # undo /5 scaling
        cur_vwap    = float(vwap_dev[-1])
        rsi_signal  = "OVERBOUGHT" if cur_rsi > 0.7 else ("OVERSOLD" if cur_rsi < 0.3 else "NEUTRAL")
        trend_arrow = "▲" if cur_vwap > 0.02 else ("▼" if cur_vwap < -0.02 else "─")

        print(f"\n{'─'*60}", flush=True)
        print(f"  📊 {sym} — {dates[0]} → {dates[-1]}  ({n} trading days)", flush=True)
        print(f"  Close:       ${last_close:>10.2f}    Total return: {ret_ytd:+.1f}%", flush=True)
        print(f"  RVol-20d:    {cur_rv20:>9.2f}%    RVol-60d:    {float(rv60[-1])*100:.2f}%", flush=True)
        print(f"  RSI-14:      {cur_rsi:>9.3f}     Signal:      {rsi_signal}", flush=True)
        print(f"  Sharpe-60d:  {cur_sharpe:>9.3f}     VWAP Dev:    {cur_vwap:+.4f} {trend_arrow}", flush=True)
        print(f"{'─'*60}\n", flush=True)

        # ── 2. Build Plotly figure ────────────────────────────────────────────
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True,
            subplot_titles=[
                f"{sym}  |  Close Price + Kalman Signal",
                "Realised Volatility  (20d  vs  60d)",
                "RSI-14  [overbought >0.7 | oversold <0.3]",
                "Rolling Sharpe-60d  (annualised, scaled)",
                "VWAP Deviation  (+ = trading above VWAP)",
            ],
            row_heights=[0.30, 0.18, 0.17, 0.17, 0.18],
            vertical_spacing=0.035,
        )

        # Panel 1 — price
        fig.add_trace(
            go.Scatter(x=dates, y=close_arr.tolist(),
                       name="Close", line=dict(color="#4E79A7", width=1.5)),
            row=1, col=1)
        # Kalman reconstructed price (close * (1 + tanh-deviation))
        kalman_price = (close_arr * (1 + kf_dev)).tolist()
        fig.add_trace(
            go.Scatter(x=dates, y=kalman_price,
                       name="Kalman", line=dict(color="#F28E2B", width=1, dash="dot")),
            row=1, col=1)

        # Panel 2 — volatility
        fig.add_trace(
            go.Scatter(x=dates, y=(rv20 * 100).tolist(),
                       name="RVol 20d %", line=dict(color="#E15759", width=1.4)),
            row=2, col=1)
        fig.add_trace(
            go.Scatter(x=dates, y=(rv60 * 100).tolist(),
                       name="RVol 60d %", line=dict(color="#B07AA1", width=1, dash="dash")),
            row=2, col=1)

        # Panel 3 — RSI with shaded OB/OS bands
        fig.add_trace(
            go.Scatter(x=dates, y=rsi14.tolist(),
                       name="RSI-14", line=dict(color="#59A14F", width=1.4)),
            row=3, col=1)
        fig.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(231,76,60,0.12)",
                      line_width=0, row=3, col=1)
        fig.add_hrect(y0=0.0, y1=0.3, fillcolor="rgba(39,174,96,0.12)",
                      line_width=0, row=3, col=1)
        fig.add_hline(y=0.7, line_dash="dot", line_color="rgba(231,76,60,0.6)",
                      row=3, col=1)
        fig.add_hline(y=0.3, line_dash="dot", line_color="rgba(39,174,96,0.6)",
                      row=3, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(180,180,180,0.4)",
                      row=3, col=1)

        # Panel 4 — Sharpe (undo /5 scaling applied during feature engineering)
        sharpe_unscaled = (sharpe60 * 5).tolist()
        fig.add_trace(
            go.Scatter(x=dates, y=sharpe_unscaled,
                       name="Sharpe-60d", line=dict(color="#76B7B2", width=1.4)),
            row=4, col=1)
        fig.add_hline(y=0, line_dash="dot",
                      line_color="rgba(180,180,180,0.5)", row=4, col=1)
        fig.add_hline(y=1, line_dash="dot",
                      line_color="rgba(89,161,79,0.5)",   row=4, col=1)

        # Panel 5 — VWAP deviation bar (green = above, red = below)
        bar_colors = ["#59A14F" if v >= 0 else "#E15759" for v in vwap_dev]
        fig.add_trace(
            go.Bar(x=dates, y=vwap_dev.tolist(), name="VWAP Dev",
                   marker_color=bar_colors, marker_line_width=0),
            row=5, col=1)
        fig.add_hline(y=0, line_dash="dot",
                      line_color="rgba(180,180,180,0.5)", row=5, col=1)

        # Layout
        fig.update_layout(
            title=dict(
                text=f"<b>{sym}</b> — Quant Stats Dashboard  "
                     f"({dates[0]} → {dates[-1]})  "
                     f"| Close ${last_close:.2f}  Return {ret_ytd:+.1f}%",
                font=dict(size=14)),
            template="plotly_dark",
            height=950,
            showlegend=True,
            legend=dict(orientation="h", y=1.03, x=0),
            margin=dict(t=80, l=60, r=20, b=40),
        )
        # Only show x-axis labels on bottom panel
        for row in range(1, 5):
            fig.update_xaxes(showticklabels=False, row=row, col=1)
        fig.update_xaxes(showticklabels=True, tickangle=-30, row=5, col=1)

        # ── 3. Show inline (Kaggle / Jupyter) ────────────────────────────────
        # fig.show() triggers the iframe renderer in Kaggle notebooks.
        # In a plain script it opens a browser tab — still useful.

        # ── 4. Save HTML ─────────────────────────────────────────────────────
        out_path = os.path.join(stock_dir, f"{sym}_stats.html")
        pio.write_html(fig, out_path, include_plotlyjs=True)
        print(f"  [Chart] {sym} → {out_path}", flush=True)

    except Exception as e:
        ts(f"  [Chart ERROR] {sym}: {e}\n{traceback.format_exc()}")



# ─────────────────────────────────────────────────────────────────────────────
# Numba kernels  (unchanged from v8 — already correct & fast)
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def _rolling_std_nb(arr, w, out):
    n = len(arr)
    for i in range(w - 1, n):
        m = 0.0
        for j in range(i - w + 1, i + 1): m += arr[j]
        m /= w; s = 0.0
        for j in range(i - w + 1, i + 1): s += (arr[j] - m) ** 2
        out[i] = math.sqrt(s / w)

@njit(cache=True)
def _rolling_mean_nb(arr, w, out):
    n = len(arr)
    for i in range(w - 1, n):
        s = 0.0
        for j in range(i - w + 1, i + 1): s += arr[j]
        out[i] = s / w

@njit(cache=True)
def _rolling_max_nb(arr, w, out):
    n = len(arr)
    for i in range(w - 1, n):
        mx = arr[i - w + 1]
        for j in range(i - w + 2, i + 1):
            if arr[j] > mx: mx = arr[j]
        out[i] = mx

@njit(cache=True)
def _rolling_min_nb(arr, w, out):
    n = len(arr)
    for i in range(w - 1, n):
        mn = arr[i - w + 1]
        for j in range(i - w + 2, i + 1):
            if arr[j] < mn: mn = arr[j]
        out[i] = mn

@njit(cache=True)
def _rolling_cov_nb(x, y, w, out):
    n = len(x)
    sx  = np.empty(n + 1, np.float64); sx[0]  = 0.0
    sy  = np.empty(n + 1, np.float64); sy[0]  = 0.0
    sxy = np.empty(n + 1, np.float64); sxy[0] = 0.0
    for i in range(n):
        sx[i+1]  = sx[i]  + x[i]
        sy[i+1]  = sy[i]  + y[i]
        sxy[i+1] = sxy[i] + x[i] * y[i]
    for i in range(w - 1, n):
        wx = sx[i+1] - sx[i-w+1]; wy = sy[i+1] - sy[i-w+1]
        wxy = sxy[i+1] - sxy[i-w+1]
        out[i] = (wxy - wx * wy / w) / (w - 1)

@njit(cache=True)
def _rolling_corr_nb(x, y, w, out):
    n = len(x)
    for i in range(w - 1, n):
        mx = 0.0; my = 0.0
        for j in range(i - w + 1, i + 1): mx += x[j]; my += y[j]
        mx /= w; my /= w; num = 0.0; dx = 0.0; dy = 0.0
        for j in range(i - w + 1, i + 1):
            a = x[j] - mx; b = y[j] - my
            num += a * b; dx += a * a; dy += b * b
        d = math.sqrt(dx * dy)
        out[i] = num / d if d > 1e-12 else 0.0

@njit(cache=True)
def _winsorize_nb(arr, q_lo, q_hi):
    for i in range(len(arr)):
        if arr[i] < q_lo:   arr[i] = q_lo
        elif arr[i] > q_hi: arr[i] = q_hi

@njit(cache=True)
def _ema_nb(arr, span, out):
    alpha = 2.0 / (span + 1.0); out[0] = arr[0]
    for i in range(1, len(arr)): out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]

@njit(cache=True)
def _rsi_nb(arr, w, out):
    n = len(arr)
    for i in range(w, n):
        gains = 0.0; losses = 0.0
        for j in range(i - w + 1, i + 1):
            d = arr[j] - arr[j - 1]
            if d > 0: gains += d
            else:     losses -= d
        avg_g = gains / w; avg_l = losses / w
        out[i] = 1.0 if avg_l < 1e-12 else 1.0 - 1.0 / (1.0 + avg_g / avg_l)

@njit(cache=True)
def _adx_nb(high, low, close, w, plus_out, minus_out, adx_out):
    n = len(close)
    tr = np.empty(n, np.float64); pdm = np.empty(n, np.float64); ndm = np.empty(n, np.float64)
    tr[0] = 0.0; pdm[0] = 0.0; ndm[0] = 0.0
    for i in range(1, n):
        hl = high[i] - low[i]; hc = abs(high[i] - close[i-1]); lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
        up = high[i] - high[i-1]; down = low[i-1] - low[i]
        pdm[i] = up   if (up > down and up > 0)   else 0.0
        ndm[i] = down if (down > up and down > 0) else 0.0
    atr_s = np.empty(n, np.float64); pdm_s = np.empty(n, np.float64); ndm_s = np.empty(n, np.float64)
    atr_s[0] = tr[0]; pdm_s[0] = pdm[0]; ndm_s[0] = ndm[0]; alpha = 1.0 / w
    for i in range(1, n):
        atr_s[i] = atr_s[i-1]*(1-alpha) + tr[i]*alpha
        pdm_s[i] = pdm_s[i-1]*(1-alpha) + pdm[i]*alpha
        ndm_s[i] = ndm_s[i-1]*(1-alpha) + ndm[i]*alpha
    dx = np.empty(n, np.float64)
    for i in range(n):
        a = atr_s[i]
        pdi = pdm_s[i]/a if a > 1e-12 else 0.0
        ndi = ndm_s[i]/a if a > 1e-12 else 0.0
        s = pdi + ndi; dx[i] = abs(pdi - ndi) / s if s > 1e-12 else 0.0
        plus_out[i] = pdi; minus_out[i] = ndi
    adx_s = np.empty(n, np.float64); adx_s[0] = dx[0]
    for i in range(1, n): adx_s[i] = adx_s[i-1]*(1-alpha) + dx[i]*alpha
    for i in range(n): adx_out[i] = adx_s[i]

@njit(cache=True)
def _obv_nb(close, volume, out):
    n = len(close); out[0] = volume[0]
    for i in range(1, n):
        if   close[i] > close[i-1]: out[i] = out[i-1] + volume[i]
        elif close[i] < close[i-1]: out[i] = out[i-1] - volume[i]
        else:                        out[i] = out[i-1]

@njit(cache=True)
def _mfi_nb(high, low, close, volume, w, out):
    n = len(close); tp = (high + low + close) / 3.0; rmf = tp * volume
    for i in range(w, n):
        pos = 0.0; neg = 0.0
        for j in range(i - w + 1, i + 1):
            if tp[j] > tp[j-1]: pos += rmf[j]
            else:                neg += rmf[j]
        out[i] = pos / (pos + neg) if (pos + neg) > 1e-12 else 0.5

@njit(cache=True)
def _cmf_nb(high, low, close, volume, w, out):
    n = len(close)
    for i in range(w - 1, n):
        sv = 0.0; tv = 0.0
        for j in range(i - w + 1, i + 1):
            hl = high[j] - low[j]
            mf = ((close[j] - low[j]) - (high[j] - close[j])) / hl if hl > 1e-12 else 0.0
            sv += mf * volume[j]; tv += volume[j]
        out[i] = sv / tv if tv > 1e-12 else 0.0

@njit(cache=True)
def _ad_nb(high, low, close, volume, out):
    n = len(close); out[0] = 0.0
    for i in range(1, n):
        hl = high[i] - low[i]
        clv = ((close[i] - low[i]) - (high[i] - close[i])) / hl if hl > 1e-12 else 0.0
        out[i] = out[i-1] + clv * volume[i]

@njit(cache=True)
def _ulcer_nb(close, w, out):
    n = len(close)
    for i in range(w - 1, n):
        pk = close[i - w + 1]
        for j in range(i - w + 2, i + 1):
            if close[j] > pk: pk = close[j]
        s = 0.0
        for j in range(i - w + 1, i + 1):
            dd = (close[j] - pk) / pk * 100.0 if pk > 1e-12 else 0.0; s += dd * dd
        out[i] = math.sqrt(s / w)

@njit(cache=True)
def _rolling_percentile_nb(arr, w, pct, out):
    n = len(arr)
    for i in range(w - 1, n):
        buf = arr[i - w + 1: i + 1].copy(); nb = len(buf)
        for ii in range(nb - 1):
            for jj in range(ii + 1, nb):
                if buf[jj] < buf[ii]: tmp = buf[ii]; buf[ii] = buf[jj]; buf[jj] = tmp
        idx = pct * (nb - 1); lo2 = int(idx); hi2 = lo2 + 1
        if hi2 >= nb: out[i] = buf[nb - 1]
        else:         out[i] = buf[lo2] + (idx - lo2) * (buf[hi2] - buf[lo2])

@njit(cache=True)
def _rolling_entropy_nb(arr, w, out):
    n = len(arr)
    for i in range(w - 1, n):
        mn = 0.0; mx = 0.0; first = True
        for j in range(i - w + 1, i + 1):
            if first or arr[j] < mn: mn = arr[j]
            if first or arr[j] > mx: mx = arr[j]
            first = False
        rng = mx - mn
        if rng < 1e-12: out[i] = 0.0; continue
        bins = 10; counts = np.zeros(bins, np.float64)
        for j in range(i - w + 1, i + 1):
            b = int((arr[j] - mn) / rng * bins)
            if b >= bins: b = bins - 1
            counts[b] += 1.0
        ent = 0.0
        for k in range(bins):
            p = counts[k] / w
            if p > 1e-12: ent -= p * math.log(p)
        out[i] = ent

@njit(cache=True)
def _kalman_filter_nb(arr, q, r, out_filtered, out_velocity):
    n = len(arr); x = arr[0]; v = 0.0; p = 1.0
    for i in range(n):
        x_pred = x + v; p_pred = p + q
        k = p_pred / (p_pred + r)
        x = x_pred + k * (arr[i] - x_pred); p = (1 - k) * p_pred
        if i > 0: v = 0.8 * v + 0.2 * (x - out_filtered[i-1])
        out_filtered[i] = x; out_velocity[i] = v

@njit(cache=True)
def _hurst_rs_nb(arr, out, window=60):
    n = len(arr)
    for i in range(window - 1, n):
        w = arr[i - window + 1: i + 1]; N = len(w)
        if N < 10: out[i] = 0.5; continue
        mn = 0.0
        for v in w: mn += v
        mn /= N; cumdev = np.empty(N, np.float64); cs = 0.0
        for j in range(N): cs += w[j] - mn; cumdev[j] = cs
        R = np.max(cumdev) - np.min(cumdev)
        S = 0.0
        for v in w: S += (v - mn) ** 2
        S = math.sqrt(S / N)
        if S < 1e-12: out[i] = 0.5; continue
        out[i] = math.log(R / S) / math.log(N) if R / S > 0 else 0.5

@njit(cache=True)
def _fractal_dim_nb(arr, w, out):
    n = len(arr)
    for i in range(w - 1, n):
        seg = arr[i - w + 1: i + 1]; N = len(seg)
        if N < 4: out[i] = 1.5; continue
        mn = np.min(seg); mx = np.max(seg)
        if mx - mn < 1e-12: out[i] = 1.0; continue
        norm = (seg - mn) / (mx - mn)
        L = 0.0
        for j in range(1, N): L += abs(norm[j] - norm[j-1])
        out[i] = 1.0 + math.log(L) / math.log(N - 1) if L > 0 else 1.0

@njit(cache=True)
def _omega_ratio_nb(arr, w, threshold, out):
    n = len(arr)
    for i in range(w - 1, n):
        seg = arr[i - w + 1: i + 1]; gains = 0.0; losses = 0.0
        for v in seg:
            if v > threshold: gains  += v - threshold
            else:             losses += threshold - v
        out[i] = gains / losses if losses > 1e-12 else 10.0

@njit(cache=True)
def _amihud_lambda_nb(ret, dvol, w, out):
    n = len(ret)
    for i in range(w - 1, n):
        num = 0.0; den = 0.0
        for j in range(i - w + 1, i + 1): num += abs(ret[j]); den += dvol[j]
        out[i] = num / den if den > 1e-12 else 0.0

@njit(cache=True)
def _kyle_lambda_nb(ret, volume, w, out):
    n = len(ret)
    for i in range(w - 1, n):
        sx = 0.0; sy = 0.0; sxy = 0.0; sx2 = 0.0
        for j in range(i - w + 1, i + 1):
            x = math.sqrt(volume[j]); sx += x; sy += ret[j]; sxy += x * ret[j]; sx2 += x * x
        mn = w; denom = sx2 * mn - sx * sx
        out[i] = (sxy * mn - sx * sy) / denom if abs(denom) > 1e-12 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Numpy / Pandas helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_div(a, b, fill=0.0):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(b) > EPS, a / b, fill)
    return np.nan_to_num(r, nan=fill, posinf=fill, neginf=fill)

def _safe_divide(a, b, fill=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) > 1e-12, a / b, fill)
    return out.astype(np.float32)

def _rstd(arr, w):
    out = np.full(len(arr), np.nan, np.float32)
    _rolling_std_nb(arr.astype(np.float64), w, out); return out

def _rmean(arr, w):
    out = np.full(len(arr), np.nan, np.float32)
    _rolling_mean_nb(arr.astype(np.float64), w, out); return out

def _rmax(arr, w):
    out = np.full(len(arr), np.nan, np.float32)
    _rolling_max_nb(arr.astype(np.float64), w, out); return out

def _rmin(arr, w):
    out = np.full(len(arr), np.nan, np.float32)
    _rolling_min_nb(arr.astype(np.float64), w, out); return out

def _rcov(x, y, w, n):
    out = np.full(n, np.nan, np.float32)
    _rolling_cov_nb(x.astype(np.float64), y.astype(np.float64), w, out); return out

def _rsi_arr(arr, w):
    out = np.full(len(arr), np.nan, np.float32)
    _rsi_nb(arr.astype(np.float64), w, out); return out

def _ema(arr, span):
    out = np.empty(len(arr), np.float32)
    _ema_nb(arr.astype(np.float64), float(span), out); return out

def rolling_mean_pd(x, w):
    return (pd.Series(x.astype(np.float64))
            .rolling(w, min_periods=max(1, w // 2)).mean()
            .bfill().ffill().fillna(0.0).values)

def rolling_std_pd(x, w):
    return (pd.Series(x.astype(np.float64))
            .rolling(w, min_periods=max(1, w // 2)).std()
            .bfill().ffill().fillna(0.0).values)

def ema_pd(x, span):
    return pd.Series(x.astype(np.float64)).ewm(span=span, adjust=False).mean().values

def winsorize_1d(x, pct=WINSOR_PCT):
    lo = np.nanpercentile(x, pct); hi = np.nanpercentile(x, 100. - pct)
    return np.clip(x, lo, hi)

def robust_zscore(x, w):
    mu = rolling_mean_pd(x, w); sigma = rolling_std_pd(x, w)
    z = safe_div(x - mu, sigma + EPS)
    return np.tanh(np.clip(z, -10., 10.))

def make_calendar_features(dates):
    n = len(dates); out = np.zeros((n, 10), dtype=np.float32)
    dow = np.array(dates.dayofweek, dtype=np.float32)
    mth = np.array(dates.month, dtype=np.float32)
    dom = np.array(dates.day, dtype=np.float32)
    qtr = np.array(dates.quarter, dtype=np.float32)
    woy = np.array(dates.isocalendar().week.values, dtype=np.float32)
    out[:, 0] = np.sin(2 * np.pi * dow / 5);  out[:, 1] = np.cos(2 * np.pi * dow / 5)
    out[:, 2] = np.sin(2 * np.pi * mth / 12); out[:, 3] = np.cos(2 * np.pi * mth / 12)
    out[:, 4] = np.sin(2 * np.pi * woy / 52); out[:, 5] = np.cos(2 * np.pi * woy / 52)
    out[:, 6] = (qtr - 1) / 3.0; out[:, 7] = (dom - 1) / 30.0
    s = pd.Series(dates)
    out[:, 8] = s.dt.is_month_end.values.astype(np.float32)
    out[:, 9] = s.dt.is_month_start.values.astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────
def _spectral_entropy(ret, w=60, nperseg=32):
    n = len(ret); out = np.zeros(n, np.float32)
    for i in range(w - 1, n):
        seg = ret[i - w + 1: i + 1].astype(np.float64)
        if np.std(seg) < 1e-10: continue
        try:
            _, psd = welch(seg, nperseg=min(nperseg, len(seg) // 2))
            psd = psd + 1e-12; psd /= psd.sum()
            out[i] = float(-np.sum(psd * np.log(psd)))
        except: pass
    return out

def _rolling_beta_decay(ret, mr, fast=20, slow=60, n=None):
    if n is None: n = len(ret)
    cov_f = np.full(n, np.nan, np.float32); cov_s = np.full(n, np.nan, np.float32)
    var_f = np.full(n, np.nan, np.float32); var_s = np.full(n, np.nan, np.float32)
    _rolling_cov_nb(ret.astype(np.float64), mr.astype(np.float64), fast, cov_f)
    _rolling_cov_nb(ret.astype(np.float64), mr.astype(np.float64), slow, cov_s)
    _rolling_std_nb(mr.astype(np.float64), fast, var_f)
    _rolling_std_nb(mr.astype(np.float64), slow, var_s)
    b_f = safe_div(cov_f.astype(np.float64), (var_f.astype(np.float64) ** 2 + EPS))
    b_s = safe_div(cov_s.astype(np.float64), (var_s.astype(np.float64) ** 2 + EPS))
    decay = safe_div(b_f - b_s, np.abs(b_s) + EPS)
    return np.tanh(np.clip(decay, -3., 3.)).astype(np.float32)

def _rolling_sharpe_nb_wrapper(ret, w=60):
    n = len(ret); out = np.zeros(n, np.float32)
    for i in range(w - 1, n):
        seg = ret[i - w + 1: i + 1]; mu = np.mean(seg); sigma = np.std(seg)
        out[i] = np.clip(mu / (sigma + EPS) * math.sqrt(252), -5., 5.)
    return out

def _rolling_sortino_nb_wrapper(ret, w=60):
    n = len(ret); out = np.zeros(n, np.float32)
    for i in range(w - 1, n):
        seg = ret[i - w + 1: i + 1]; mu = np.mean(seg)
        ds = seg[seg < 0]; sigma_d = np.std(ds) if len(ds) > 1 else EPS
        out[i] = np.clip(mu / (sigma_d + EPS) * math.sqrt(252), -5., 5.)
    return out

def _rolling_calmar_nb_wrapper(ret, w=60):
    n = len(ret); out = np.zeros(n, np.float32)
    for i in range(w - 1, n):
        seg = ret[i - w + 1: i + 1]; mu = np.mean(seg) * 252
        cum = np.cumprod(1 + np.clip(seg, -0.5, 0.5))
        rm = np.maximum.accumulate(cum)
        dd = np.abs(np.min((cum - rm) / (rm + EPS)))
        out[i] = np.clip(mu / (dd + EPS), -10., 10.)
    return out

def _intraday_range_efficiency(high, low, close, w=20):
    n = len(close); out = np.zeros(n, np.float32)
    for i in range(w - 1, n):
        net = abs(close[i] - close[i - w + 1])
        path = np.sum(high[i - w + 1: i + 1] - low[i - w + 1: i + 1])
        out[i] = float(safe_div(np.array([net]), np.array([path + EPS]))[0])
    return out

def _volume_weighted_momentum(ret, volume, w=20):
    vw_ret = ret * volume; vw_ma = rolling_mean_pd(vw_ret, w)
    vol_ma = rolling_mean_pd(volume, w)
    return np.tanh(safe_div(vw_ma, vol_ma + EPS)).astype(np.float32)

def _realized_skewness_risk(ret, w=60):
    n = len(ret); out = np.zeros(n, np.float32)
    for i in range(w - 1, n):
        seg = ret[i - w + 1: i + 1]
        if len(seg) < 10: continue
        try: out[i] = np.clip(float(scipy_stats.skew(seg)), -5., 5.)
        except: pass
    return out

def _price_momentum_oscillator(close, fast=12, slow=26, signal=9):
    ema_f = ema_pd(close, fast); ema_s = ema_pd(close, slow)
    ppo = (ema_f - ema_s) / (np.abs(ema_s) + EPS) * 100.
    ppo_sig = ema_pd(ppo, signal)
    return np.tanh((ppo - ppo_sig) / 5.).astype(np.float32)

def _stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    n = len(close); k = np.zeros(n, np.float32)
    for i in range(k_period - 1, n):
        hi = np.max(high[i - k_period + 1: i + 1])
        lo = np.min(low[i - k_period + 1: i + 1])
        k[i] = float(safe_div(np.array([close[i] - lo]), np.array([hi - lo + EPS]))[0])
    d = rolling_mean_pd(k, d_period)
    return (k - 0.5).astype(np.float32), (d - 0.5).astype(np.float32)

def _williams_r(high, low, close, w=14):
    n = len(close); out = np.zeros(n, np.float32)
    for i in range(w - 1, n):
        hi = np.max(high[i - w + 1: i + 1]); lo = np.min(low[i - w + 1: i + 1])
        out[i] = float(safe_div(np.array([hi - close[i]]), np.array([hi - lo + EPS]))[0])
    return (out - 0.5).astype(np.float32)

def _linear_regression_slope(arr, w=20):
    n = len(arr); out = np.zeros(n, np.float32)
    x = np.arange(w, dtype=np.float64); xm = x.mean(); xd = ((x - xm) ** 2).sum()
    for i in range(w - 1, n):
        y = arr[i - w + 1: i + 1].astype(np.float64); yd = (y * (x - xm)).sum()
        slope = yd / xd if xd > 1e-12 else 0.
        out[i] = float(np.clip(slope / (np.std(y) + EPS), -5., 5.))
    return out

def _realized_variance_ratio(ret, short_w=5, long_w=20):
    rv_s = rolling_std_pd(ret, short_w); rv_l = rolling_std_pd(ret, long_w)
    ratio = safe_div(rv_s ** 2 * (long_w / short_w), rv_l ** 2 + EPS)
    return np.tanh(ratio - 1.).astype(np.float32)

def _price_level_features(close, w_list=(5, 10, 20, 50, 100, 200)):
    features = {}
    for w in w_list:
        ma = rolling_mean_pd(close, w)
        features[f"price_to_ma{w}"] = np.tanh(safe_div(close - ma, np.abs(ma) + EPS)).astype(np.float32)
    return features

def _flow_toxicity_vpin(close, volume, w=50):
    n = len(close); ret = np.zeros(n, np.float64)
    ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + EPS)
    sigma = rolling_std_pd(ret, w)
    cdf_vals = scipy_stats.norm.cdf(ret / (sigma + EPS))
    buy_vol = volume * cdf_vals; sell_vol = volume * (1 - cdf_vals)
    buy_ma = rolling_mean_pd(buy_vol, w); sell_ma = rolling_mean_pd(sell_vol, w)
    total_ma = rolling_mean_pd(volume.astype(np.float64), w)
    vpin = safe_div(np.abs(buy_ma - sell_ma), total_ma + EPS)
    return np.tanh(robust_zscore(vpin, w * 2)).astype(np.float32)

def _microstructure_noise(close, high, low, w=20):
    hl = (high - low).astype(np.float64)
    oc = np.abs(close - np.roll(close, 1)).astype(np.float64); oc[0] = 0.
    noise = safe_div(oc, hl + EPS)
    noise_ma = rolling_mean_pd(noise, w)
    return np.tanh(np.clip(robust_zscore(noise_ma, w * 3), -3., 3.)).astype(np.float32)

# ── NEW v9: multi-scale entropy ────────────────────────────────────────────
def _multi_scale_entropy(ret, scales=(5, 20, 60)):
    """Entropy at multiple lookback scales → regime complexity signal."""
    n = len(ret); outs = []
    for w in scales:
        ent = np.full(n, np.nan, np.float32)
        _rolling_entropy_nb(ret.astype(np.float64), w, ent)
        outs.append(np.nan_to_num(ent))
    # cross-scale difference: long-term entropy minus short-term
    diff = outs[-1] - outs[0]
    return outs[0], outs[1], outs[2], diff

# ── NEW v9: cross-sectional z-score of rolling returns ────────────────────
def _cs_zscore(arr):
    mu = np.nanmean(arr); sigma = np.nanstd(arr)
    return float(np.clip((arr - mu) / (sigma + EPS), -5., 5.))


# ─────────────────────────────────────────────────────────────────────────────
# Core feature computation (causal — no lookahead by construction)
# ─────────────────────────────────────────────────────────────────────────────
def process_symbol(df, mr, mv, mb,
                   sym_daily_ranks=None,
                   sector_ret_series=None,
                   sector_code=None,
                   exchange_code=None):
    """
    Returns (mat: float32 [T, F], feat_names: List[str]).
    All features use only past/present data at each time t.
    """
    n = len(df); f = {}
    close  = df["close"].values.astype(np.float32)
    open_  = df["open"].values.astype(np.float32)
    high   = df["high"].values.astype(np.float32)
    low    = df["low"].values.astype(np.float32)
    volume = df["volume"].values.astype(np.float32)
    dvol   = df["dollar_volume"].values.astype(np.float32)
    ema20  = df["ema_20"].values.astype(np.float32)
    ema50  = df["ema_50"].values.astype(np.float32)
    ema100 = df["ema_100"].values.astype(np.float32)
    atr14  = df["atr_14"].values.astype(np.float32)
    mr = mr.astype(np.float32); mv = mv.astype(np.float32); mb = mb.astype(np.float32)

    # ── 1-day return (base series) ──────────────────────────────────────────
    ret1 = np.empty(n, np.float32); ret1[0] = np.nan
    ret1[1:] = _safe_divide(close[1:] - close[:-1], close[:-1])
    r64 = ret1.astype(np.float64)

    # ── Multi-horizon past returns (no lookahead) ───────────────────────────
    for d in [1, 2, 3, 5, 7, 10, 14, 20, 28, 50, 60, 100]:
        r = np.full(n, np.nan, np.float32)
        r[d:] = _safe_divide(close[d:] - close[:-d], close[:-d])
        f[f"return_{d}d"] = r

    # ── Realised volatility ─────────────────────────────────────────────────
    rv5 = _rstd(ret1, 5);   rv10  = _rstd(ret1, 10)
    rv20 = _rstd(ret1, 20); rv60  = _rstd(ret1, 60)
    rv120 = _rstd(ret1, 120); rv252 = _rstd(ret1, 252)
    f["realized_vol_5"]   = rv5;   f["realized_vol_10"]  = rv10
    f["realized_vol_20"]  = rv20;  f["realized_vol_60"]  = rv60
    f["realized_vol_120"] = rv120; f["realized_vol_252"] = rv252
    f["vol_regime"]       = _safe_divide(rv20, rv120)
    f["vol_of_vol"]       = _rstd(rv20, 20)
    f["vol_ratio_5_20"]   = _safe_divide(rv5, rv20)
    f["vol_ratio_20_60"]  = _safe_divide(rv20, rv60)
    f["vol_ratio_60_120"] = _safe_divide(rv60, rv120)

    # ── Trend / EMA ─────────────────────────────────────────────────────────
    f["trend_strength"]   = _safe_divide(ema20 - ema100, atr14)
    f["ema_ratio_20_50"]  = _safe_divide(ema20, ema50) - 1.0
    f["ema_ratio_50_100"] = _safe_divide(ema50, ema100) - 1.0
    f["close_to_ema20"]   = _safe_divide(close - ema20, atr14)
    f["close_to_ema50"]   = _safe_divide(close - ema50, atr14)
    f["close_to_ema100"]  = _safe_divide(close - ema100, atr14)
    f["atr_ratio"]        = _safe_divide(atr14, close)
    tf_base = ema20 - ema50; tf_shift = np.empty(n, np.float32)
    tf_shift[0] = 0.0; tf_shift[1:] = tf_base[:-1]
    f["trend_flip"] = (tf_base * tf_shift < 0).astype(np.float32)
    sma200 = np.full(n, np.nan, np.float32)
    _rolling_mean_nb(close.astype(np.float64), 200, sma200)
    f["price_to_sma200"] = _safe_divide(close, sma200) - 1.0

    # ── ADX ─────────────────────────────────────────────────────────────────
    plus_di = np.full(n, np.nan, np.float32)
    minus_di = np.full(n, np.nan, np.float32)
    adx_arr  = np.full(n, np.nan, np.float32)
    _adx_nb(high.astype(np.float64), low.astype(np.float64), close.astype(np.float64),
            14, plus_di, minus_di, adx_arr)
    f["adx14"] = adx_arr; f["plus_di14"] = plus_di; f["minus_di14"] = minus_di
    f["adx_strength"] = (adx_arr > 25.).astype(np.float32)

    # ── MACD ────────────────────────────────────────────────────────────────
    macd_line = _ema(close, 12) - _ema(close, 26)
    f["macd_hist"] = macd_line - _ema(macd_line, 9)
    f["macd_line"] = np.tanh(macd_line / (close + EPS))

    # ── RSI ─────────────────────────────────────────────────────────────────
    f["rsi14"] = _rsi_arr(close, 14); f["rsi7"] = _rsi_arr(close, 7)
    f["rsi21"] = _rsi_arr(close, 21); f["rsi_divergence"] = f["rsi14"] - f["rsi21"]

    # ── 52-week range ────────────────────────────────────────────────────────
    hi52 = _rmax(close, 252); lo52 = _rmin(close, 252)
    f["price_pct_52w"] = _safe_divide(close - lo52, hi52 - lo52)

    # ── Momentum ─────────────────────────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        r = np.full(n, np.nan, np.float32)
        r[w:] = _safe_divide(close[w:] - close[:-w], close[:-w])
        f[f"mom_{w}"] = r
    f["price_accel"]         = f["mom_5"] - f["mom_10"]
    f["momentum_decay"]      = _safe_divide(f["mom_5"], np.abs(f["mom_20"]) + EPS)
    f["vol_adjusted_mom_20"] = np.tanh(_safe_divide(f["mom_20"], rv60 + EPS))

    # ── Market regime features ───────────────────────────────────────────────
    mv_ma50 = _rmean(mv, 50)
    f["market_regime"]      = ((mv > mv_ma50) & (mr > 0)).astype(np.float32)
    f["regime_strength"]    = (mr * mv).astype(np.float32)
    f["market_trend"]       = _rmean(mr, 20)
    f["market_vol_trend"]   = _rmean(mv, 20)
    f["breadth_adj_market"] = (mr * mb).astype(np.float32)
    f["spy_return"]         = mr.copy(); f["spy_volatility"] = mv.copy()
    spy_rmax = _rmax(mr, 20)
    f["spy_drawdown"]  = _safe_divide(mr - spy_rmax, np.abs(spy_rmax) + 1e-9)
    f["spy_rsi"]       = _rsi_arr(mr, 14)
    f["market_breadth"] = mb.copy()
    f["advance_decline_ratio"] = _safe_divide(np.maximum(mb, 0.), np.maximum(1. - mb, 1e-9))

    # ── Volume / liquidity ───────────────────────────────────────────────────
    vm = _rmean(volume, 20); vs = _rstd(volume, 20)
    vm252 = _rmean(volume, 252); vm60 = _rmean(volume, 60)
    vsurp = _safe_divide(volume, vm, fill=1.0)
    f["vol_z20"]           = _safe_divide(volume - vm, vs)
    f["volume_zscore20"]   = f["vol_z20"]
    f["vol_surprise"]      = vsurp
    f["volume_trend"]      = _safe_divide(_rmean(volume, 5), vm)
    f["turnover_ratio"]    = _safe_divide(volume, vm252)
    f["relative_volume20"] = _safe_divide(volume, vm,   fill=1.0)
    f["relative_volume60"] = _safe_divide(volume, vm60, fill=1.0)
    df_flow = np.full(n, np.nan, np.float32)
    df_flow[5:] = _safe_divide(dvol[5:] - dvol[:-5], dvol[:-5])
    f["dollar_flow"]        = df_flow
    f["average_daily_volume"] = vm252

    # ── Amihud / Kyle illiquidity ────────────────────────────────────────────
    amihud_raw = np.full(n, np.nan, np.float32)
    _amihud_lambda_nb(np.nan_to_num(ret1).astype(np.float64), dvol.astype(np.float64), 20, amihud_raw)
    f["amihud"] = np.tanh(robust_zscore(np.nan_to_num(amihud_raw), 60)).astype(np.float32)
    kyle_raw = np.full(n, np.nan, np.float32)
    _kyle_lambda_nb(np.nan_to_num(ret1).astype(np.float64), volume.astype(np.float64), 20, kyle_raw)
    f["kyle_lambda"] = np.tanh(robust_zscore(np.nan_to_num(kyle_raw), 60)).astype(np.float32)
    f["liq_dryness"] = _safe_divide(np.ones(n, np.float32), vsurp)
    lv = np.log1p(volume).astype(np.float32)
    f["liq_trend"] = _rmean(lv, 20)
    lv_accel = np.full(n, np.nan, np.float32)
    lv_accel[5:] = lv[5:] - lv[:-5]; f["liq_accel"] = lv_accel
    vpct = np.full(n, np.nan, np.float32)
    _rolling_percentile_nb(volume.astype(np.float64), min(252, n), 0.5, vpct)
    f["volume_percentile252"] = _safe_divide(volume, vpct, fill=1.0)

    # ── OBV / MFI / CMF / A-D ────────────────────────────────────────────────
    obv_arr = np.empty(n, np.float32)
    _obv_nb(close.astype(np.float64), volume.astype(np.float64), obv_arr)
    f["obv"] = np.tanh(robust_zscore(obv_arr, 60)).astype(np.float32)
    obv_slope = np.full(n, np.nan, np.float32)
    obv_slope[5:] = _safe_divide(obv_arr[5:] - obv_arr[:-5], np.abs(obv_arr[:-5]) + 1.0)
    f["obv_slope"] = np.tanh(np.nan_to_num(obv_slope))
    mfi_out = np.full(n, np.nan, np.float32)
    _mfi_nb(high.astype(np.float64), low.astype(np.float64),
            close.astype(np.float64), volume.astype(np.float64), 14, mfi_out)
    f["money_flow_index"] = np.nan_to_num(mfi_out) - 0.5
    cmf_out = np.full(n, np.nan, np.float32)
    _cmf_nb(high.astype(np.float64), low.astype(np.float64),
            close.astype(np.float64), volume.astype(np.float64), 20, cmf_out)
    f["chaikin_money_flow"] = np.nan_to_num(cmf_out)
    ad_out = np.empty(n, np.float32)
    _ad_nb(high.astype(np.float64), low.astype(np.float64),
           close.astype(np.float64), volume.astype(np.float64), ad_out)
    f["accumulation_distribution"] = np.tanh(robust_zscore(ad_out, 60)).astype(np.float32)

    # ── Price action candle features ─────────────────────────────────────────
    hl = (high - low).clip(min=1e-9)
    f["bid_ask_proxy"] = _safe_divide(high - low, close)
    f["hl_spread"]     = _safe_divide(high - low, close)
    f["oc_return"]     = _safe_divide(close - open_, open_)
    f["body"]          = _safe_divide(close - open_, hl)
    f["close_pos"]     = _safe_divide(close - low, hl)
    upper = (high - np.maximum(open_, close)).clip(min=0.)
    lower = (np.minimum(open_, close) - low).clip(min=0.)
    f["upper_shadow"] = _safe_divide(upper, hl)
    f["lower_shadow"] = _safe_divide(lower, hl)
    gap = np.full(n, np.nan, np.float32)
    gap[1:] = _safe_divide(open_[1:] - close[:-1], close[:-1])
    f["gap"] = np.nan_to_num(gap)
    f["overnight_gap_ma"] = _rmean(np.nan_to_num(gap), 10)
    f["true_range"] = _safe_divide(atr14, close)
    body_abs = np.abs(close - open_)
    body_pct = np.full(n, np.nan, np.float32)
    _rolling_percentile_nb(body_abs.astype(np.float64), 20, 0.5, body_pct)
    f["body_percentile"] = _safe_divide(body_abs, body_pct + 1e-9)
    f["wick_ratio"] = _safe_divide(upper + lower, body_abs + 1e-9)
    inside_bar = np.zeros(n, np.float32); outside_bar = np.zeros(n, np.float32)
    for i in range(1, n):
        if high[i] < high[i-1] and low[i] > low[i-1]: inside_bar[i] = 1.
        if high[i] > high[i-1] and low[i] < low[i-1]: outside_bar[i] = 1.
    f["inside_bar"] = inside_bar; f["outside_bar"] = outside_bar

    # ── Drawdown / risk ──────────────────────────────────────────────────────
    rh = _rmax(close, 20); dd = _safe_divide(close - rh, rh)
    f["drawdown"]   = dd; f["recovery"] = _rmean(dd, 20)
    f["breakout"]   = _safe_divide(close, _rmax(high, 20))
    f["support"]    = _safe_divide(close, _rmin(low, 20))
    f["pain_index"] = _rmean(np.abs(dd), 60)
    ulcer_out = np.full(n, np.nan, np.float32)
    _ulcer_nb(close.astype(np.float64), 14, ulcer_out)
    f["ulcer_index"]   = np.nan_to_num(ulcer_out)
    f["vol_breakout"]  = _safe_divide(rv20, _rmean(rv20, 50))
    f["shock"]         = _safe_divide(np.abs(np.nan_to_num(f["return_1d"])), rv20)
    f["range_expansion"] = _safe_divide(high - low, _rmean(close, 20))
    f["jump_intensity"] = _rmean(
        (np.abs(ret1) > 2 * _rstd(ret1, 60)).astype(np.float32), 60)

    # ── Parkinson / Garman-Klass vol estimators ──────────────────────────────
    log_hl = np.log(np.maximum(high, 1e-9) / np.maximum(low,  1e-9)).astype(np.float64)
    log_co = np.log(np.maximum(close,1e-9) / np.maximum(open_,1e-9)).astype(np.float64)
    pk_raw = np.full(n, np.nan, np.float32)
    for i in range(19, n):
        s = 0.0
        for j in range(i - 19, i + 1): s += log_hl[j] ** 2
        pk_raw[i] = math.sqrt(s / (4 * 20 * math.log(2)))
    f["parkinson_vol"] = np.nan_to_num(pk_raw)
    gk_raw = np.full(n, np.nan, np.float32)
    for i in range(19, n):
        s1 = 0.0; s2 = 0.0
        for j in range(i - 19, i + 1):
            s1 += 0.5 * log_hl[j] ** 2; s2 += (2 * math.log(2) - 1) * log_co[j] ** 2
        gk_raw[i] = math.sqrt(max(0., (s1 - s2) / 20))
    f["garman_klass_vol"] = np.nan_to_num(gk_raw)

    # ── Downside / upside dev, skew, kurtosis ────────────────────────────────
    ds_dev = np.full(n, np.nan, np.float32); us_dev = np.full(n, np.nan, np.float32)
    _rolling_std_nb(np.where(ret1 < 0, ret1, 0.).astype(np.float64), 20, ds_dev)
    _rolling_std_nb(np.where(ret1 > 0, ret1, 0.).astype(np.float64), 20, us_dev)
    f["downside_dev20"] = np.nan_to_num(ds_dev); f["upside_dev20"] = np.nan_to_num(us_dev)
    f["vol_skew_proxy"] = np.tanh(
        _safe_divide(np.nan_to_num(us_dev), np.nan_to_num(ds_dev) + EPS) - 1.)
    s_ret = pd.Series(ret1)
    f["skew20"]  = s_ret.rolling(20,  min_periods=15).skew().values.astype(np.float32)
    f["kurt20"]  = s_ret.rolling(20,  min_periods=15).kurt().values.astype(np.float32)
    f["skew60"]  = s_ret.rolling(60,  min_periods=30).skew().values.astype(np.float32)

    # ── NEW: multi-scale entropy ──────────────────────────────────────────────
    ent5, ent20, ent60, ent_diff = _multi_scale_entropy(np.nan_to_num(r64))
    f["entropy_5"]    = ent5.astype(np.float32)
    f["entropy_20"]   = ent20.astype(np.float32)
    f["entropy_60"]   = ent60.astype(np.float32)
    f["entropy_diff"] = ent_diff.astype(np.float32)

    # ── Original entropy + tail ratio ────────────────────────────────────────
    ent_out = np.full(n, np.nan, np.float32)
    _rolling_entropy_nb(r64, 60, ent_out); f["rolling_entropy"] = np.nan_to_num(ent_out)
    pos_flag = (ret1 > 0).astype(np.float32); pos_flag[0] = np.nan
    f["ret_pos_frac_20"] = _rmean(np.nan_to_num(pos_flag), 20)
    p95 = np.full(n, np.nan, np.float32); p5 = np.full(n, np.nan, np.float32)
    _rolling_percentile_nb(np.abs(ret1).astype(np.float64), 60, 0.95, p95)
    _rolling_percentile_nb(np.abs(ret1).astype(np.float64), 60, 0.05, p5)
    f["tail_ratio"] = np.nan_to_num(_safe_divide(p95, p5 + 1e-9))

    # ── Correlations with SPY ────────────────────────────────────────────────
    corr20 = np.full(n, np.nan, np.float32)
    corr60 = np.full(n, np.nan, np.float32)
    _rolling_corr_nb(ret1.astype(np.float64), mr.astype(np.float64), 20, corr20)
    _rolling_corr_nb(ret1.astype(np.float64), mr.astype(np.float64), 60, corr60)
    f["correlation_spy20"] = np.nan_to_num(corr20)
    f["correlation_spy60"] = np.nan_to_num(corr60)
    f["return_vs_spy20"]   = np.nan_to_num(f["return_20d"]) - np.nan_to_num(_rmean(mr, 20) * 20)

    # ── NEW v9: 3-window covariance with market ──────────────────────────────
    for ww in [10, 20, 60]:
        cov_w = np.full(n, np.nan, np.float32)
        _rolling_cov_nb(np.nan_to_num(ret1).astype(np.float64),
                        mr.astype(np.float64), ww, cov_w)
        f[f"cov_spy_{ww}"] = np.tanh(np.nan_to_num(cov_w) * 100.)

    # ── Sharpe / Sortino / Calmar proxies ────────────────────────────────────
    rm60m = _rmean(ret1, 60)
    f["sharpe_proxy"]  = np.tanh(_safe_divide(rm60m, rv60) * math.sqrt(252) / 3.)
    f["sortino_proxy"] = np.tanh(
        _safe_divide(rm60m, np.nan_to_num(ds_dev) + 1e-9) * math.sqrt(252) / 3.)
    f["calmar_proxy"]  = np.tanh(
        _safe_divide(np.nan_to_num(f["return_20d"]), np.abs(dd) + 1e-9) / 2.)
    f["sharpe_60"]  = _rolling_sharpe_nb_wrapper(np.nan_to_num(ret1), 60) / 5.
    f["sortino_60"] = _rolling_sortino_nb_wrapper(np.nan_to_num(ret1), 60) / 5.
    f["calmar_60"]  = _rolling_calmar_nb_wrapper(np.nan_to_num(ret1), 60) / 10.

    # ── Residual / alpha ──────────────────────────────────────────────────────
    beta_arr = _safe_divide(
        _rcov(ret1, mr, 60, n),
        (rv60 ** 2 + 1e-9))
    resid = ret1 - np.nan_to_num(beta_arr) * mr
    f["residual_volatility"] = _rstd(np.nan_to_num(resid), 60)
    f["rolling_alpha"]       = np.tanh(_rmean(np.nan_to_num(resid), 60) * 252 * 10)
    f["tracking_error"]      = np.nan_to_num(_rstd(resid, 60)) * math.sqrt(252)

    # ── Volume-price trend ────────────────────────────────────────────────────
    vpt = np.full(n, np.nan, np.float32); vpt[1:] = volume[1:] * ret1[1:]
    vpt = np.tanh(robust_zscore(np.nan_to_num(vpt).clip(-1e6, 1e6), 60)).astype(np.float32)
    f["volume_price_trend"] = vpt
    f["liquidity_shock"]    = (vsurp * np.nan_to_num(f["shock"])).astype(np.float32)

    # ── VWAP ─────────────────────────────────────────────────────────────────
    tp = (high + low + close) / 3.
    cum_vol = np.cumsum(volume) + EPS; cum_tpv = np.cumsum(tp * volume)
    vwap = cum_tpv / cum_vol
    f["vwap_deviation"] = np.tanh(np.clip(_safe_divide(close - vwap, vwap + EPS), -3., 3.))

    # ── Order-flow imbalance ──────────────────────────────────────────────────
    buy_vol  = np.where(close >= open_, volume, 0.).astype(np.float64)
    sell_vol = np.where(close <  open_, volume, 0.).astype(np.float64)
    buy_ma   = rolling_mean_pd(buy_vol,  20)
    sell_ma  = rolling_mean_pd(sell_vol, 20)
    f["orderflow_imbalance"] = np.tanh(
        np.clip(safe_div(buy_ma - sell_ma, buy_ma + sell_ma + EPS), -3., 3.)).astype(np.float32)

    # ── Microstructure ────────────────────────────────────────────────────────
    f["microstructure_noise"]   = _microstructure_noise(close, high, low)
    f["flow_toxicity_vpin"]     = _flow_toxicity_vpin(close, volume)
    corr_out = np.full(n, 0., np.float32)
    _rolling_corr_nb(np.abs(ret1).astype(np.float64), volume.astype(np.float64), 20, corr_out)
    f["informed_trading_proxy"] = np.tanh(
        np.clip(corr_out.astype(np.float64), -3., 3.)).astype(np.float32)
    pi   = safe_div(np.abs(np.nan_to_num(ret1)), np.sqrt(volume + EPS))
    pi_ma = rolling_mean_pd(pi, 10)
    pi_lag = np.roll(pi_ma, 5); pi_lag[:5] = pi_ma[5]
    f["price_impact_persistence"] = np.tanh(
        np.clip(safe_div(pi_ma - pi_lag, np.abs(pi_lag) + EPS), -3., 3.)).astype(np.float32)

    # ── Beta decay / Hurst / Fractal ──────────────────────────────────────────
    f["beta_decay"] = _rolling_beta_decay(np.nan_to_num(ret1), mr)
    hurst_out = np.full(n, 0.5, np.float32)
    _hurst_rs_nb(np.nan_to_num(r64), hurst_out); f["hurst_exponent"] = hurst_out - 0.5
    fractal_out = np.full(n, 1.5, np.float32)
    _fractal_dim_nb(np.nan_to_num(ret1).astype(np.float64), 30, fractal_out)
    f["fractal_dimension"] = np.nan_to_num(fractal_out) - 1.5
    omega_out = np.full(n, 1., np.float32)
    _omega_ratio_nb(np.nan_to_num(ret1).astype(np.float64), 60, 0., omega_out)
    f["omega_ratio"] = np.tanh((np.nan_to_num(omega_out) - 1.) / 2.)
    f["spectral_entropy"]   = _spectral_entropy(np.nan_to_num(ret1)).astype(np.float32)
    f["realized_skew_risk"] = _realized_skewness_risk(np.nan_to_num(ret1))
    f["var_ratio"]          = _realized_variance_ratio(np.nan_to_num(ret1))

    # ── Kalman filter ─────────────────────────────────────────────────────────
    kf_filt = np.empty(n, np.float64); kf_vel = np.empty(n, np.float64)
    _kalman_filter_nb(close.astype(np.float64), 0.01, 1.0, kf_filt, kf_vel)
    f["kalman_filtered_price"] = np.tanh(
        safe_div(close - kf_filt, np.abs(kf_filt) + EPS)).astype(np.float32)
    f["kalman_velocity"] = np.tanh(kf_vel * 100.).astype(np.float32)

    # ── Technical oscillators ─────────────────────────────────────────────────
    f["pmo"] = _price_momentum_oscillator(close.astype(np.float64))
    stoch_k, stoch_d = _stochastic_oscillator(high, low, close)
    f["stoch_k"] = stoch_k; f["stoch_d"] = stoch_d
    f["stoch_kd_diff"] = stoch_k - stoch_d
    f["williams_r"]    = _williams_r(high, low, close)
    f["lr_slope_20"]   = _linear_regression_slope(close.astype(np.float64), 20)
    f["lr_slope_5"]    = _linear_regression_slope(close.astype(np.float64), 5)
    f["intraday_range_efficiency"] = _intraday_range_efficiency(high, low, close)
    f["volume_weighted_mom"]  = _volume_weighted_momentum(np.nan_to_num(ret1), volume)
    f["vwm_fast"]             = _volume_weighted_momentum(np.nan_to_num(ret1), volume)
    f["momentum_quality"] = np.tanh(np.clip(
        rolling_mean_pd(np.sign(ret1).astype(float), 20) *
        rolling_mean_pd(np.abs(ret1), 20) * 100., -5., 5.)).astype(np.float32)

    # ── Price level features ──────────────────────────────────────────────────
    pl = _price_level_features(close.astype(np.float64), [5, 10, 20, 50, 100, 200])
    for k, v in pl.items(): f[k] = v

    # ── Cross-sectional rank ──────────────────────────────────────────────────
    if sym_daily_ranks is not None:
        f["cs_return_rank"]          = np.nan_to_num(sym_daily_ranks).astype(np.float32)
        f["market_percentile_return"] = f["cs_return_rank"]
    else:
        f["market_percentile_return"] = np.zeros(n, np.float32)

    # ── Sector features ───────────────────────────────────────────────────────
    if sector_ret_series is not None:
        sec_r20 = np.full(n, np.nan, np.float32)
        _rolling_mean_nb(sector_ret_series.values.astype(np.float64), 20, sec_r20)
        f["return_vs_sector20"] = np.nan_to_num(f["return_20d"]) - np.nan_to_num(sec_r20)
        sec_r60 = np.full(n, np.nan, np.float32)
        _rolling_mean_nb(sector_ret_series.values.astype(np.float64), 60, sec_r60)
        f["sector_momentum_60"] = np.nan_to_num(sec_r60)
    else:
        f["return_vs_sector20"] = np.zeros(n, np.float32)
        f["sector_momentum_60"] = np.zeros(n, np.float32)

    # ── Calendar ──────────────────────────────────────────────────────────────
    dates_idx = df.index
    dow = np.array(pd.DatetimeIndex(dates_idx).dayofweek, dtype=np.float32)
    mth = np.array(pd.DatetimeIndex(dates_idx).month,     dtype=np.float32)
    qtr = np.array(pd.DatetimeIndex(dates_idx).quarter,   dtype=np.float32)
    woy = np.array(pd.DatetimeIndex(dates_idx).isocalendar().week.values, dtype=np.float32)
    dom = np.array(pd.DatetimeIndex(dates_idx).day,       dtype=np.float32)
    s_dt = pd.Series(pd.DatetimeIndex(dates_idx))
    f["day_of_week_sin"]  = np.sin(2 * np.pi * dow / 5)
    f["day_of_week_cos"]  = np.cos(2 * np.pi * dow / 5)
    f["month_sin"]        = np.sin(2 * np.pi * mth / 12)
    f["month_cos"]        = np.cos(2 * np.pi * mth / 12)
    f["week_sin"]         = np.sin(2 * np.pi * woy / 52)
    f["week_cos"]         = np.cos(2 * np.pi * woy / 52)
    f["month_end"]        = s_dt.dt.is_month_end.values.astype(np.float32)
    f["month_start"]      = s_dt.dt.is_month_start.values.astype(np.float32)
    f["quarter"]          = (qtr - 1) / 3.
    f["dom_norm"]         = (dom - 1) / 30.

    # ── Metadata embeddings ───────────────────────────────────────────────────
    f["sector"]   = np.full(n, float(sector_code)   / 20. if sector_code   is not None else 0., np.float32)
    f["exchange"] = np.full(n, float(exchange_code) / 5.  if exchange_code is not None else 0., np.float32)

    # ── Assemble ──────────────────────────────────────────────────────────────
    feat_names = sorted(f.keys())
    mat = np.stack([f[k] for k in feat_names], axis=1).astype(np.float32)
    mat = np.nan_to_num(mat, nan=0., posinf=0., neginf=0.)
    # Winsorise per column (1–99 pct) to prevent outlier contamination
    for j in range(mat.shape[1]):
        col = mat[:, j]; finite = col[np.isfinite(col)]
        if len(finite) > 10:
            lo, hi2 = np.percentile(finite, 1.), np.percentile(finite, 99.)
            _winsorize_nb(col, lo, hi2); mat[:, j] = col
    np.round(mat, 4, out=mat)
    return mat, feat_names


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder — no leakage, strict causal slicing
# ─────────────────────────────────────────────────────────────────────────────
def make_sequences(mat, cal, seq_len, step=1):
    n = mat.shape[0]
    if n < seq_len:
        return (np.empty((0, seq_len, mat.shape[1]), dtype=np.float16),
                np.empty((0, seq_len, cal.shape[1]), dtype=np.float16),
                np.empty(0, dtype=np.int32))
    mat = np.nan_to_num(mat, nan=0., posinf=0., neginf=0.)
    mat = np.clip(mat, -FLOAT16_MAX, FLOAT16_MAX)
    idx = np.arange(seq_len - 1, n, step, dtype=np.int32)
    out_u = np.empty((len(idx), seq_len, mat.shape[1]), dtype=np.float16)
    out_k = np.empty((len(idx), seq_len, cal.shape[1]), dtype=np.float16)
    for i, end in enumerate(idx):
        sl = slice(end - seq_len + 1, end + 1)
        out_u[i] = mat[sl].astype(np.float16)
        out_k[i] = cal[sl].astype(np.float16)
    return out_u, out_k, idx


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Parquet → V4 (chunked, memory-safe, cleanup after each ticker)
# ─────────────────────────────────────────────────────────────────────────────
def execute_stage1_parquet_to_v4():
    os.makedirs(OUT_DIR_V4, exist_ok=True)
    ts("Stage1: scanning input directory")
    if not os.path.exists(BASE_INPUT_DIR):
        ts(f"WARN: {BASE_INPUT_DIR} not found — synthetic mock mode"); return []

    files = [f for f in os.listdir(BASE_INPUT_DIR) if f.endswith(".parquet")]
    if not files: ts("WARN: no parquet files found"); return []

    ts(f"Stage1: loading {len(files)} parquet files in chunks")
    # Load one file at a time to avoid peak RAM spike
    frames = []
    for fn in files:
        tmp = pd.read_parquet(os.path.join(BASE_INPUT_DIR, fn))
        frames.append(tmp)
    raw = pd.concat(frames, ignore_index=True); del frames; gc.collect()

    raw = raw.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="first")
    raw.drop(columns=[c for c in DROP_RAW_COLUMNS if c in raw.columns], inplace=True)
    for c in raw.select_dtypes("float64").columns: raw[c] = raw[c].astype("float32")
    for c in raw.select_dtypes("int64").columns:   raw[c] = raw[c].astype("int32")
    ts(f"Stage1: {len(raw)} rows, {raw['symbol'].nunique()} symbols loaded")

    # Cross-sectional return rank (using only past data via groupby-date then rank)
    raw["cs_return_rank"] = (
        raw.assign(ret1=raw.groupby("symbol")["close"].pct_change())
           .groupby("date")["ret1"].rank(pct=True))

    sector_codes   = {}; exchange_codes = {}; sym_sector_map = {}
    if "sector" in raw.columns:
        sc = raw[["symbol", "sector"]].drop_duplicates("symbol")
        cat_s = pd.Categorical(sc["sector"])
        for sym, code in zip(sc["symbol"], cat_s.codes): sector_codes[sym] = int(code)
        sym_sector_map = (raw[["symbol", "sector"]].drop_duplicates("symbol")
                             .set_index("symbol")["sector"])
    if "exchange" in raw.columns:
        ec = raw[["symbol", "exchange"]].drop_duplicates("symbol")
        cat_e = pd.Categorical(ec["exchange"])
        for sym, code in zip(ec["symbol"], cat_e.codes): exchange_codes[sym] = int(code)

    market_df = (raw[["date", "market_return", "market_volatility", "market_breadth"]]
                 .drop_duplicates("date").set_index("date").sort_index())

    sector_daily_ret = None
    if "sector" in raw.columns:
        tmp2 = raw[["date", "symbol", "sector", "close"]].copy()
        tmp2["ret1"] = tmp2.groupby("symbol")["close"].pct_change()
        sector_daily_ret = tmp2.groupby(["date", "sector"])["ret1"].mean().unstack("sector")
        del tmp2; gc.collect()

    feat_names_saved = None; processed_syms = []; n_skipped = 0
    symbols = raw["symbol"].unique(); total_syms = len(symbols)
    stage1_start = time.time()
    chart_out = CFG.out_dir   # charts written here (created later, but mkdir is safe)

    for i, sym in enumerate(symbols):
        sym_start = time.time()
        # ── Guard rails ───────────────────────────────────────────────────
        ram_gb = _check_ram_gb()
        store_gb = _check_storage_gb(OUT_DIR_V4)
        if ram_gb > RAM_LIMIT_GB:
            ts(f"[Stage1] RAM limit {RAM_LIMIT_GB}GB hit at symbol [{i+1}/{total_syms}] "
               f"{sym} (RAM={ram_gb:.1f}GB) — stopping stage1"); break
        if store_gb > STORAGE_LIMIT_GB:
            ts(f"[Stage1] Storage limit {STORAGE_LIMIT_GB}GB hit at symbol "
               f"[{i+1}/{total_syms}] {sym} — stopping stage1"); break

        # ── Print EVERY stock being processed ─────────────────────────────
        done = (i + 1) / total_syms
        print(f"  [Stage1 {i+1}/{total_syms}] Processing {sym} | "
              f"RAM={ram_gb:.1f}GB | Disk={store_gb:.1f}GB | "
              f"ETA={eta(done, stage1_start)}", flush=True)

        sdf = raw[raw["symbol"] == sym].set_index("date").sort_index()
        min_rows = SEQ_LEN + max(HORIZONS) + 10
        if len(sdf) < min_rows:
            print(f"  [Stage1] SKIP {sym}: only {len(sdf)} rows "
                  f"(need {min_rows})", flush=True)
            n_skipped += 1; continue

        try:
            mr_arr = market_df.loc[sdf.index, "market_return"].values.astype(np.float32)
            mv_arr = market_df.loc[sdf.index, "market_volatility"].values.astype(np.float32)
            mb_arr = market_df.loc[sdf.index, "market_breadth"].values.astype(np.float32)
            cs_rank_arr = sdf["cs_return_rank"].values if "cs_return_rank" in sdf.columns else None
            sec_ret_for_sym = None
            if (sector_daily_ret is not None and
                    sym in getattr(sym_sector_map, "index", [])):
                sym_sec = sym_sector_map[sym]
                if sym_sec in sector_daily_ret.columns:
                    sec_ret_for_sym = (sector_daily_ret.loc[sdf.index, sym_sec]
                                       .fillna(0.).values.astype(np.float32))

            s_code = sector_codes.get(sym); e_code = exchange_codes.get(sym)
            print(f"    → computing features for {sym} ({len(sdf)} rows)...", flush=True)
            mat, feat_names = process_symbol(
                sdf, mr_arr, mv_arr, mb_arr, cs_rank_arr,
                sector_ret_series=(pd.Series(sec_ret_for_sym, index=sdf.index)
                                   if sec_ret_for_sym is not None else None),
                sector_code=s_code, exchange_code=e_code)

            print(f"    → building sequences for {sym}...", flush=True)
            cal = make_calendar_features(pd.DatetimeIndex(sdf.index))
            seqs_u, seqs_k, end_idx = make_sequences(mat, cal, SEQ_LEN, STEP)
            if seqs_u.shape[0] == 0:
                print(f"    → SKIP {sym}: 0 sequences produced", flush=True)
                continue

            # ── Disk pre-check before writing ─────────────────────────────
            if not _check_free_before_write(OUT_DIR_V4, needed_gb=0.3):
                ts(f"[Stage1] Aborting {sym} write — disk too full"); n_skipped += 1; continue

            raw_close = sdf["close"].values.astype(np.float32)
            print(f"    → saving {seqs_u.shape[0]} sequences for {sym} "
                  f"shape={seqs_u.shape}...", flush=True)
            np.save(os.path.join(OUT_DIR_V4, f"{sym}_close.npy"),   raw_close)
            np.save(os.path.join(OUT_DIR_V4, f"{sym}_endidx.npy"),  end_idx)
            np.save(os.path.join(OUT_DIR_V4, f"{sym}_unknown.npy"), seqs_u)
            np.save(os.path.join(OUT_DIR_V4, f"{sym}_known.npy"),   seqs_k)

            if feat_names_saved is None:
                feat_names_saved = feat_names
                with open(os.path.join(OUT_DIR_V4, "feature_names_unknown.txt"), "w") as fh:
                    fh.write("\n".join(feat_names))
                with open(os.path.join(OUT_DIR_V4, "feature_names_known.txt"), "w") as fh:
                    fh.write("\n".join(CALENDAR_NAMES))
                ts(f"Stage1: {len(feat_names)} features computed (first stock)")

            # ── Per-stock chart (every stock) ──────────────────────────────
            os.makedirs(chart_out, exist_ok=True)
            save_stock_chart(sym, sdf, mat, feat_names, chart_out)
            if (i + 1) % 10 == 0:
                ts(f"    → chart saved for {sym}")

            elapsed_sym = time.time() - sym_start
            print(f"  [Stage1] ✓ {sym} done in {elapsed_sym:.1f}s | "
                  f"seqs={seqs_u.shape[0]} feats={mat.shape[1]}", flush=True)
            processed_syms.append(sym)

        except Exception as e:
            ts(f"[Stage1] ERROR on {sym}: {e}")
            ts(f"[Stage1] Traceback:\n{traceback.format_exc()}")
            n_skipped += 1
        finally:
            try: del sdf, mat, cal, seqs_u, seqs_k, mr_arr, mv_arr, mb_arr
            except: pass
            if i % 10 == 0: gc.collect()

    del raw, market_df; gc.collect()
    ts(f"Stage1 done: {len(processed_syms)} processed, {n_skipped} skipped")
    return processed_syms


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: V4 → V5 (normalise + cleanup V4 per ticker)
# ─────────────────────────────────────────────────────────────────────────────
def execute_stage2_v4_to_v5(tickers):
    os.makedirs(OUT_DIR_V5, exist_ok=True)
    ts(f"Stage2: normalising {len(tickers)} tickers → {OUT_DIR_V5}")

    uf = os.path.join(OUT_DIR_V4, "feature_names_unknown.txt")
    kf = os.path.join(OUT_DIR_V4, "feature_names_known.txt")
    if not os.path.exists(uf): ts("WARN: feature names not found"); return

    with open(uf) as fh: unk_names = [l.strip() for l in fh]
    with open(kf) as fh: kn_names  = [l.strip() for l in fh]
    ts(f"Stage2: {len(unk_names)} unknown features, {len(kn_names)} known features")

    # ── Compute global stats from a sample (TRAIN PORTION ONLY) ──────────────
    # We approximate train portion as first ~75% of each ticker's sequences.
    global_mu = None; global_sigma = None; sample_count = 0
    sample_syms = tickers[:min(40, len(tickers))]
    ts("Stage2: computing global normalisation stats from sample (train rows only)")
    for ticker in sample_syms:
        p = os.path.join(OUT_DIR_V4, f"{ticker}_unknown.npy")
        if not os.path.exists(p): continue
        try:
            X = np.load(p, mmap_mode="r").astype(np.float32)
            S = X.shape[0]
            # Only use the train portion
            t_end = int(S * (1 - CFG.test_frac - CFG.val_frac))
            if t_end < 10: t_end = max(10, int(S * 0.7))
            flat = X[:t_end].reshape(-1, X.shape[-1])
            if global_mu is None:
                global_mu    = np.nanmean(flat, axis=0)
                global_sigma = np.nanstd(flat,  axis=0)
                sample_count = flat.shape[0]
            else:
                n1 = sample_count; n2 = flat.shape[0]; mu2 = np.nanmean(flat, axis=0)
                global_mu    = (global_mu * n1 + mu2 * n2) / (n1 + n2)
                global_sigma = np.sqrt(
                    (global_sigma ** 2 * n1 + np.nanvar(flat, axis=0) * n2) / (n1 + n2))
                sample_count += n2
            del X, flat; gc.collect()
        except Exception as e:
            ts(f"  WARN norm-stats {ticker}: {e}")

    if global_mu is None:
        global_mu    = np.zeros(len(unk_names), np.float32)
        global_sigma = np.ones(len(unk_names),  np.float32)
    global_sigma = np.where(global_sigma < EPS, 1., global_sigma)
    np.save(os.path.join(OUT_DIR_V5, "global_mu.npy"),    global_mu)
    np.save(os.path.join(OUT_DIR_V5, "global_sigma.npy"), global_sigma)

    stage2_start = time.time()
    for ti, ticker in enumerate(tickers):
        ram_gb  = _check_ram_gb()
        free_gb = _free_space_gb("/tmp")
        if ram_gb > RAM_LIMIT_GB:
            ts(f"[Stage2] RAM limit {RAM_LIMIT_GB}GB hit at ticker "
               f"[{ti+1}/{len(tickers)}] {ticker} (RAM={ram_gb:.1f}GB)"); break

        # ── Print EVERY ticker ─────────────────────────────────────────────
        done = (ti + 1) / len(tickers)
        print(f"  [Stage2 {ti+1}/{len(tickers)}] Normalising {ticker} | "
              f"RAM={ram_gb:.1f}GB | FreeGB={free_gb:.1f} | "
              f"ETA={eta(done, stage2_start)}", flush=True)

        p_unk = os.path.join(OUT_DIR_V4, f"{ticker}_unknown.npy")
        p_kn  = os.path.join(OUT_DIR_V4, f"{ticker}_known.npy")
        p_cl  = os.path.join(OUT_DIR_V4, f"{ticker}_close.npy")
        p_ei  = os.path.join(OUT_DIR_V4, f"{ticker}_endidx.npy")

        if not (os.path.exists(p_unk) and os.path.exists(p_kn)):
            print(f"  [Stage2] SKIP {ticker}: v4 files not found "
                  f"(unk={os.path.exists(p_unk)} kn={os.path.exists(p_kn)})", flush=True)
            continue

        try:
            print(f"    → loading v4 for {ticker}...", flush=True)
            X_unk = np.load(p_unk).astype(np.float32)
            X_kn  = np.load(p_kn).astype(np.float32)

            # ── Copy close/endidx to v5 FIRST (before we delete v4) ───────
            for tag, src_p in [("close", p_cl), ("endidx", p_ei)]:
                dst = os.path.join(OUT_DIR_V5, f"{ticker}_{tag}.npy")
                if os.path.exists(src_p):
                    shutil.copy2(src_p, dst)

            # ── KEY FIX: delete v4 files BEFORE writing v5 to free space ──
            print(f"    → deleting v4 for {ticker} to free disk...", flush=True)
            _safe_delete(p_unk, p_kn, p_cl, p_ei)
            gc.collect()

            # ── Disk pre-check ─────────────────────────────────────────────
            if not _check_free_before_write("/tmp", needed_gb=0.2):
                ts(f"[Stage2] Aborting {ticker} — disk too full even after v4 delete")
                del X_unk, X_kn; gc.collect(); continue

            print(f"    → normalising {ticker} shape={X_unk.shape}...", flush=True)
            X_f   = np.clip((X_unk - global_mu) / (global_sigma + EPS), -5., 5.)
            X_f   = np.nan_to_num(X_f,   nan=0., posinf=0., neginf=0.)
            X_kn_f = np.nan_to_num(np.clip(X_kn, -1., 1.), nan=0.)

            print(f"    → saving v5 for {ticker}...", flush=True)
            np.save(os.path.join(OUT_DIR_V5, f"{ticker}_unknown.npy"), X_f)
            np.save(os.path.join(OUT_DIR_V5, f"{ticker}_known.npy"),   X_kn_f)

            # Write feature names once
            fn_u = os.path.join(OUT_DIR_V5, "feature_names_unknown.txt")
            if not os.path.exists(fn_u):
                with open(fn_u, "w") as fh: fh.write("\n".join(unk_names))
                with open(os.path.join(OUT_DIR_V5, "feature_names_known.txt"), "w") as fh:
                    fh.write("\n".join(kn_names))
                ts(f"Stage2: feature name files written")

            print(f"  [Stage2] ✓ {ticker} normalised OK | "
                  f"FreeGB={_free_space_gb('/tmp'):.1f}", flush=True)

        except Exception as e:
            ts(f"[Stage2] ERROR on {ticker}: {e}")
            ts(f"[Stage2] Traceback:\n{traceback.format_exc()}")
            # Still try to clean up v4 to free space
            _safe_delete(p_unk, p_kn, p_cl, p_ei)
        finally:
            try: del X_unk, X_kn, X_f, X_kn_f
            except: pass
            if ti % 10 == 0: gc.collect()

    # Remove the v4 directory entirely
    try: shutil.rmtree(OUT_DIR_V4, ignore_errors=True)
    except: pass
    ts("Stage2 done — v4 directory cleaned up")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry-point
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    ts("=== PIPELINE START ===")
    syms = execute_stage1_parquet_to_v4()
    if not syms:
        ts("No real data found — running synthetic mock pipeline")
        syms = ["MOCK"]; os.makedirs(OUT_DIR_V4, exist_ok=True)
        n_feats = TARGET_N_FEATURES; T_raw = SEQ_LEN + max(HORIZONS) + 20
        mock_close = np.cumprod(1 + np.random.randn(T_raw) * 0.01).astype(np.float32)
        mock_u = np.random.randn(25, SEQ_LEN, n_feats).astype(np.float16)
        mock_k = np.random.randn(25, SEQ_LEN, 10).astype(np.float16)
        end_idx = np.arange(SEQ_LEN - 1, SEQ_LEN - 1 + 25 * STEP, STEP, dtype=np.int32)
        np.save(os.path.join(OUT_DIR_V4, "MOCK_unknown.npy"), mock_u)
        np.save(os.path.join(OUT_DIR_V4, "MOCK_known.npy"),   mock_k)
        np.save(os.path.join(OUT_DIR_V4, "MOCK_close.npy"),   mock_close)
        np.save(os.path.join(OUT_DIR_V4, "MOCK_endidx.npy"),  end_idx)
        names = [f"f{i}" for i in range(n_feats)]
        names[0] = "return_1d"; names[1] = "realized_vol_20"; names[2] = "realized_vol_5"
        names[3] = "realized_vol_60"; names[4] = "rsi14"; names[5] = "mom_20"
        with open(os.path.join(OUT_DIR_V4, "feature_names_unknown.txt"), "w") as fh:
            fh.write("\n".join(names))
        with open(os.path.join(OUT_DIR_V4, "feature_names_known.txt"), "w") as fh:
            fh.write("\n".join(CALENDAR_NAMES))
    execute_stage2_v4_to_v5(syms)
    ts("=== PIPELINE COMPLETE ===")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
class CFG:
    data_dir   = OUT_DIR_V5
    out_dir    = "/kaggle/working/tft_v9_output"
    hidden     = 256
    heads      = 8
    n_lstm_layers = 2
    dropout    = 0.12
    n_experts  = 8           # v9: 8 experts (was 6)
    top_k_moe  = 3           # v9: top-3 (was top-2)
    enc_len    = 120
    horizons   = HORIZONS
    quantiles  = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    epochs     = 80
    batch      = 48
    lr         = 2.5e-4
    min_lr     = 5e-7
    weight_decay  = 6e-5
    grad_clip     = 1.0
    patience      = 22
    warmup_epochs = 8
    grad_accum    = 6          # v9: more accumulation for stability
    lambda_q      = 0.32
    lambda_sign   = 0.10
    lambda_ic     = 0.10
    lambda_ndcg   = 0.08
    lambda_sharpe = 0.05
    lambda_contrastive = 0.03  # v9: new temporal-contrastive regulariser
    label_smooth  = 0.02
    mixup_alpha   = 0.15
    cutmix_prob   = 0.08
    use_sam       = True
    sam_rho       = 0.05
    swa_start_frac = 0.70
    tta_passes    = 5
    amp_enabled   = True
    grad_ckpt     = True
    num_workers   = 0
    max_tickers   = None
    val_frac      = 0.15
    test_frac     = 0.10
    seed          = 42
    purge_gap     = PURGE_GAP  # rows between train/val and val/test
    drop_features = {"ipo_age", "fwd_vol_ratio", "fwd_return"}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def get_device():
    if not torch.cuda.is_available():
        ts("CPU mode — no CUDA"); return torch.device("cpu"), False, 1
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        ts(f"GPU {i}: {name} | {vram:.1f}GB VRAM")
    if n_gpus >= 2: CFG.batch = 64; ts(f"Multi-GPU: batch scaled to {CFG.batch}")
    return torch.device("cuda"), CFG.amp_enabled, n_gpus


# ─────────────────────────────────────────────────────────────────────────────
# SAM optimiser
# ─────────────────────────────────────────────────────────────────────────────
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups; self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        gn = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (gn + EPS)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        dev = self.param_groups[0]["params"][0].device
        return torch.stack([
            p.grad.norm(2).to(dev) for g in self.param_groups
            for p in g["params"] if p.grad is not None]).norm(2)

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self.base_optimizer.param_groups = self.param_groups


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — purge-gap aware, no leakage
# ─────────────────────────────────────────────────────────────────────────────
class RawDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        npy  = list(self.data_dir.glob("*.npy"))
        unk  = {f.stem.replace("_unknown", "") for f in npy if f.stem.endswith("_unknown")}
        kn   = {f.stem.replace("_known",   "") for f in npy if f.stem.endswith("_known")}
        self.tickers = sorted(unk & kn)
        if CFG.max_tickers: self.tickers = self.tickers[:CFG.max_tickers]
        ts(f"DataLoader: found {len(self.tickers)} tickers")

        uf = self.data_dir / "feature_names_unknown.txt"
        kf = self.data_dir / "feature_names_known.txt"
        self.unk_names = uf.read_text().strip().split("\n") if uf.exists() else []
        self.kn_names  = kf.read_text().strip().split("\n") if kf.exists() else []

        keep = [n not in CFG.drop_features for n in self.unk_names]
        self.unk_names = [n for n, k in zip(self.unk_names, keep) if k]
        self.keep_idx  = np.where(keep)[0]
        self.n_unk = len(self.unk_names); self.n_kn = len(self.kn_names)
        ts(f"DataLoader: {self.n_unk} unknown features, {self.n_kn} known features")

        # Feature group assignment
        GROUPS = {
            "momentum":      ["mom_","return_","macd","price_accel","momentum","pmo","lr_slope"],
            "volatility":    ["realized_vol","vol_","garman","parkinson","atr","fractal","spectral","var_ratio"],
            "trend":         ["ema_ratio","adx","trend_","hurst","close_to_ema","price_to_ma","price_to_sma","kalman"],
            "mean_rev":      ["rsi","stoch","williams","autocorr","rolling_entropy","omega","entropy_"],
            "volume":        ["obv","volume_","dollar_flow","amihud","kyle","liq_","mfi","chaikin",
                              "accumulation","money_flow","vpt","relative_volume","average_daily",
                              "vwap","orderflow","flow_toxicity","vwm"],
            "risk":          ["sharpe","sortino","calmar","tail","drawdown","recovery","pain",
                              "ulcer","downside","upside","tracking","rolling_alpha","residual","omega_ratio"],
            "price_act":     ["body","hl_spread","gap","breakout","inside","outside","lower_shadow",
                              "upper_shadow","range_expansion","oc_return","true_range","jump","shock","wick","intraday"],
            "market":        ["market_","spy_","breadth","advance_decline","cs_return","regime","sector",
                              "cov_spy_"],
            "calendar":      ["day_of_week","month_","week_","quarter","month_end","month_start","dom_norm"],
            "microstructure":["microstructure","informed_trading","price_impact","beta_decay",
                              "realized_skew","vol_skew"],
            "other":         [],
        }
        gnames = list(GROUPS.keys())
        self.feat_group_ids = []
        for name in self.unk_names:
            assigned = len(gnames) - 1
            for gi, (gn, prefixes) in enumerate(GROUPS.items()):
                if any(name.startswith(p) or p in name for p in prefixes):
                    assigned = gi; break
            self.feat_group_ids.append(assigned)
        self.n_groups = len(gnames)
        ts(f"DataLoader: {self.n_groups} feature groups")

    def load_ticker(self, ticker):
        try:
            X_unk = np.load(self.data_dir / f"{ticker}_unknown.npy", mmap_mode="r")
            X_kn  = np.load(self.data_dir / f"{ticker}_known.npy",   mmap_mode="r")
            X_unk = X_unk[:, :, self.keep_idx]
            cl_p  = self.data_dir / f"{ticker}_close.npy"
            ei_p  = self.data_dir / f"{ticker}_endidx.npy"
            if not (cl_p.exists() and ei_p.exists()): return None
            raw_close = np.load(cl_p,  mmap_mode="r")
            end_idx   = np.load(ei_p,  mmap_mode="r")
            return {"ticker": ticker, "X_unk": X_unk, "X_kn": X_kn,
                    "raw_close": raw_close, "end_idx": end_idx}
        except Exception as e:
            ts(f"WARN: load {ticker}: {e}"); return None


class MultiHorizonDataset(Dataset):
    """
    Leakage-free dataset.

    Split boundaries use a PURGE GAP equal to max(horizon) between sets
    to prevent any lookahead contamination of labels across splits.

    Target labels: log(close[ei + h] / close[ei])  — strictly future.
    The decoder known-features (dec_kn) contain ONLY calendar features
    for the future horizon window — no price or volume data.
    """
    def __init__(self, raw_loader, split="train", norm_stats=None,
                 mixup_alpha=0., cutmix_prob=0.):
        self.loader     = raw_loader; self.split = split; self.norm_stats = norm_stats
        self.mixup_alpha = mixup_alpha; self.cutmix_prob = cutmix_prob
        self.enc_len    = CFG.enc_len; self.horizons = CFG.horizons
        self.max_h      = max(self.horizons); self.n_unk = raw_loader.n_unk
        self.n_kn       = raw_loader.n_kn; self.n_h = len(self.horizons)
        self.purge      = CFG.purge_gap

        try:    self.ret_idx = raw_loader.unk_names.index("return_1d")
        except: self.ret_idx = 0

        ts(f"Dataset[{split}]: building index map...")
        self.ticker_data = []; self.index_map = []; n_static = None
        build_start = time.time()

        for ti, ticker in enumerate(raw_loader.tickers):
            if (ti + 1) % 10 == 0 or ti == 0 or ti == len(raw_loader.tickers) - 1:
                done = (ti + 1) / len(raw_loader.tickers)
                print(f"  [Dataset-{split}] {ti+1}/{len(raw_loader.tickers)} tickers | "
                      f"{len(self.index_map):,} windows so far | "
                      f"ETA={eta(done, build_start)}", flush=True)
            d = raw_loader.load_ticker(ticker)
            if d is None: continue

            X_unk = d["X_unk"]; X_kn = d["X_kn"]
            raw_close = d["raw_close"]; end_idx = d["end_idx"]
            S, T, _ = X_unk.shape; T_raw = len(raw_close)

            # Build forward-return labels — strictly no lookahead
            fwd_ret = np.full((S, self.n_h), np.nan, dtype=np.float32)
            valid   = np.zeros(S, dtype=bool)
            for s in range(S):
                ei = int(end_idx[s])
                if ei + self.max_h >= T_raw: continue
                ok = True
                for hi, h in enumerate(self.horizons):
                    if (ei + h < T_raw and raw_close[ei] > 0 and raw_close[ei + h] > 0):
                        fwd_ret[s, hi] = math.log(
                            float(raw_close[ei + h]) / float(raw_close[ei]))
                    else:
                        ok = False; break
                if ok: valid[s] = True

            if not valid.any(): continue

            # ── Purge-gap split boundaries ────────────────────────────────
            t_test = int(S * (1 - CFG.test_frac))
            t_val  = int(S * (1 - CFG.test_frac - CFG.val_frac))
            t_val  = max(1, t_val)
            t_test = max(t_val + 1, t_test)
            # Purge: skip PURGE_GAP samples at each boundary
            train_end = t_val - self.purge
            val_start = t_val + self.purge
            val_end   = t_test - self.purge
            test_start = t_test + self.purge

            sv = np.zeros(3, np.float32)
            if n_static is None: n_static = len(sv)
            tidx = len(self.ticker_data)
            self.ticker_data.append((X_unk, X_kn, raw_close, end_idx, fwd_ret, sv))

            for s in range(S):
                if not valid[s]: continue
                if   split == "train" and 0 <= s < train_end:
                    self.index_map.append((tidx, s))
                elif split == "val"   and val_start <= s < val_end:
                    self.index_map.append((tidx, s))
                elif split == "test"  and s >= test_start:
                    self.index_map.append((tidx, s))

        print(flush=True)
        self.n_static = n_static or 3
        ts(f"Dataset[{split}]: {len(self.index_map):,} windows from "
           f"{len(self.ticker_data)} tickers")

    def compute_norm_stats(self):
        """Compute normalisation stats from TRAIN split only — no val/test leakage."""
        ts("Computing normalisation stats from train split...")
        n = min(8000, len(self.index_map))
        idxs = np.random.choice(len(self.index_map), n, replace=False)
        rows = []; tgt_rows = []
        for i in idxs:
            tidx, s = self.index_map[i]
            X_unk, _, _, _, fwd_ret, _ = self.ticker_data[tidx]
            enc = np.array(X_unk[s, :self.enc_len, :], dtype=np.float32)
            np.nan_to_num(enc, copy=False, nan=0.)
            rows.append(enc.reshape(-1, self.n_unk)); tgt_rows.append(fwd_ret[s])
        sample = np.concatenate(rows)
        mu    = np.nanmean(sample, axis=0).astype(np.float32)
        sigma = np.nanstd(sample,  axis=0).astype(np.float32)
        sigma = np.where(sigma < EPS, 1., sigma)
        tgt_flat = np.concatenate(tgt_rows)
        tgt_mu   = float(np.nanmean(tgt_flat))
        tgt_sigma = max(float(np.nanstd(tgt_flat)), EPS)
        ts(f"Norm stats: target mu={tgt_mu:.5f} sigma={tgt_sigma:.5f}")
        return {"unk_mu": mu, "unk_sigma": sigma, "tgt_mu": tgt_mu, "tgt_sigma": tgt_sigma}

    def apply_norm(self, stats): self.norm_stats = stats

    def __len__(self): return len(self.index_map)

    def __getitem__(self, idx):
        tidx, s = self.index_map[idx]
        X_unk, X_kn, raw_close, end_idx, fwd_ret, sv = self.ticker_data[tidx]

        enc_unk = np.array(X_unk[s, :self.enc_len, :], dtype=np.float32)
        enc_kn  = np.array(X_kn[s, :self.enc_len, :], dtype=np.float32)
        # Decoder known: ONLY calendar features for the future window
        # We use zeros for unknown features to avoid future leakage
        dec_kn  = np.zeros((self.max_h, self.n_kn), dtype=np.float32)
        # Fill in calendar (known) features if the known features have future calendar rows
        # Since X_kn only covers up to seq_len, we leave dec_kn as zeros (safe)
        target  = fwd_ret[s].copy()

        if self.norm_stats is not None:
            mu, sig = self.norm_stats["unk_mu"], self.norm_stats["unk_sigma"]
            enc_unk = np.clip((enc_unk - mu) / sig, -5., 5.)

        # Mixup augmentation (train only)
        if self.split == "train" and self.mixup_alpha > 0 and random.random() < 0.4:
            idx2 = random.randint(0, len(self.index_map) - 1)
            tidx2, s2 = self.index_map[idx2]
            X2, _, _, _, fwd2, _ = self.ticker_data[tidx2]
            enc2 = np.array(X2[s2, :self.enc_len, :], dtype=np.float32)
            if self.norm_stats is not None:
                enc2 = np.clip((enc2 - mu) / sig, -5., 5.)
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            enc_unk = lam * enc_unk + (1 - lam) * enc2
            target  = lam * target  + (1 - lam) * fwd2[s2]

        # CutMix augmentation (train only)
        if self.split == "train" and random.random() < self.cutmix_prob:
            cut = random.randint(1, max(1, self.enc_len // 8))
            cs  = random.randint(0, self.enc_len - cut)
            idx2 = random.randint(0, len(self.index_map) - 1)
            tidx2, s2 = self.index_map[idx2]
            X2, *_ = self.ticker_data[tidx2]
            enc2 = np.array(X2[s2, :self.enc_len, :], dtype=np.float32)
            if self.norm_stats is not None:
                enc2 = np.clip((enc2 - mu) / sig, -5., 5.)
            enc_unk[cs:cs + cut] = enc2[cs:cs + cut]

        for arr in [enc_unk, enc_kn, dec_kn]:
            np.nan_to_num(arr, copy=False, nan=0., posinf=0., neginf=0.)
        target = np.nan_to_num(target, nan=0., posinf=0., neginf=0.)

        # Static features: vol, mean return, annualised return (from encoder only)
        static_feat = np.array([
            np.mean(np.abs(enc_unk[:, self.ret_idx])),
            np.std(enc_unk[:, self.ret_idx]),
            np.mean(enc_unk[:, self.ret_idx]) * 252,
        ], dtype=np.float32)
        static_feat = np.clip(static_feat, -5., 5.)

        return (torch.from_numpy(enc_unk),
                torch.from_numpy(enc_kn),
                torch.from_numpy(dec_kn),
                torch.from_numpy(target),
                torch.from_numpy(static_feat))


def fast_collate(batch):
    eu, ek, dk, tgt, sv = zip(*batch)
    return (torch.stack(eu), torch.stack(ek), torch.stack(dk),
            torch.stack(tgt), torch.stack(sv))


# ─────────────────────────────────────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """Instance normalisation reversible — computed on encoder window only."""
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__(); self.eps = eps; self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta  = nn.Parameter(torch.zeros(n_features))
        self._mean = self._std = None

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self._mean = x.mean(1, keepdim=True).detach()
            self._std  = (x.std(1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            if self.affine: x = x * self.gamma + self.beta
        elif mode == "denorm":
            if self.affine: x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self._std + self._mean
        return x


class ALiBi(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        slopes = torch.tensor(
            [2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)], dtype=torch.float32)
        self.register_buffer("slopes", slopes)

    def forward(self, q_len, k_len, device):
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)
        dist  = (q_pos - k_pos).abs().float()
        bias  = -self.slopes.view(-1, 1, 1) * dist.unsqueeze(0)
        return bias.unsqueeze(0)


class RoPESelfAttn(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        inv = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv); self._build(max_len)

    def _build(self, L):
        t = torch.arange(L, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_c", emb.cos()[None, None, :, :])
        self.register_buffer("sin_c", emb.sin()[None, None, :, :])

    @staticmethod
    def _rot_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        L = q.shape[2]
        if L > self.cos_c.shape[2]: self._build(L)
        cos = self.cos_c[:, :, :L, :q.shape[-1]]
        sin = self.sin_c[:, :, :L, :q.shape[-1]]
        return q * cos + self._rot_half(q) * sin, k * cos + self._rot_half(k) * sin


class GLU(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.):
        super().__init__()
        self.fc   = nn.Linear(d_in, d_out)
        self.gate = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x): return self.drop(self.fc(x)) * torch.sigmoid(self.gate(x))


class GRN(nn.Module):
    def __init__(self, d_model, d_hidden=None, d_ctx=None, d_out=None, dropout=0.1):
        super().__init__()
        d_hidden = d_hidden or d_model; d_out = d_out or d_model
        self.fc1  = nn.Linear(d_model + (d_ctx or 0), d_hidden)
        self.fc2  = nn.Linear(d_hidden, d_out)
        self.elu  = nn.ELU()
        self.gate = GLU(d_out, d_out, dropout)
        self.norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_model, d_out) if d_model != d_out else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.ones(d_out) * 0.1)

    def forward(self, x, ctx=None):
        res = self.skip(x)
        inp = torch.cat([x, ctx], dim=-1) if ctx is not None else x
        h   = self.elu(self.fc1(inp)); h = self.drop(self.fc2(h)); h = self.gate(h)
        return self.norm(self.layer_scale * h + res)


class VSN(nn.Module):
    def __init__(self, n_vars, d_model, d_ctx=None, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars; self.d_model = d_model
        self.var_grns   = nn.ModuleList([GRN(d_model, dropout=dropout) for _ in range(n_vars)])
        self.weight_grn = GRN(n_vars * d_model, d_hidden=d_model, d_ctx=d_ctx, d_out=n_vars, dropout=dropout)
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, embeddings, ctx=None):
        stk  = torch.stack(embeddings, dim=-2)
        flat = stk.reshape(*stk.shape[:-2], self.n_vars * self.d_model)
        w    = self.softmax(self.weight_grn(flat, ctx))
        lead = stk.shape[:-2]
        var_outs = [grn(stk[..., i, :].reshape(-1, self.d_model)).reshape(*lead, self.d_model)
                    for i, grn in enumerate(self.var_grns)]
        var_stk  = torch.stack(var_outs, dim=-2)
        combined = (var_stk * w.unsqueeze(-1)).sum(-2)
        return combined, w


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=512):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads; self.d_head = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)
        self.qkv     = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)
        self.rope    = RoPESelfAttn(self.d_head, max_len)
        self.norm    = nn.LayerNorm(d_model); self._attn_w = None

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def sh(t): return t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q, K, V = sh(q), sh(k), sh(v); Q, K = self.rope(Q, K)
        sc = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None: sc = sc.masked_fill(mask, float("-inf"))
        attn = self.drop(F.softmax(sc, dim=-1)); self._attn_w = attn.detach()
        out  = (attn @ V).transpose(1, 2).reshape(B, T, -1)
        return self.norm(self.out_proj(out) + x)


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads; self.d_head = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)
        self.alibi   = ALiBi(n_heads)
        self.norm    = nn.LayerNorm(d_model); self._attn_w = None

    def forward(self, query, key, value=None):
        if value is None: value = key
        B, Tq, _ = query.shape; _, Tk, _ = key.shape
        def sh(t, Ts): return t.reshape(B, Ts, self.n_heads, self.d_head).transpose(1, 2)
        Q = sh(self.q_proj(query), Tq)
        K = sh(self.k_proj(key),   Tk)
        V = sh(self.v_proj(value), Tk)
        bias = self.alibi(Tq, Tk, query.device)
        sc   = (Q @ K.transpose(-2, -1)) / self.scale + bias
        attn = self.drop(F.softmax(sc, dim=-1)); self._attn_w = attn.detach()
        out  = (attn @ V).transpose(1, 2).reshape(B, Tq, -1)
        return self.norm(self.out_proj(out) + query)


class TCNBlock(nn.Module):
    def __init__(self, d_model, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad  = ((kernel - 1) * dilation) // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel, padding=pad, dilation=dilation, groups=d_model)
        self.pw   = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))
        h = self.drop(F.gelu(self.pw(h)))
        return self.norm(h.transpose(1, 2) + x)


class DilatedTCN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            TCNBlock(d_model, kernel=3, dilation=1,  dropout=dropout),
            TCNBlock(d_model, kernel=3, dilation=2,  dropout=dropout),
            TCNBlock(d_model, kernel=5, dilation=1,  dropout=dropout),
            TCNBlock(d_model, kernel=5, dilation=4,  dropout=dropout),
            TCNBlock(d_model, kernel=7, dilation=1,  dropout=dropout),   # v9: extra scale
        )

    def forward(self, x): return self.blocks(x)


class MoEFFN(nn.Module):
    """Mixture-of-Experts FFN with load-balancing auxiliary loss."""
    def __init__(self, d_model, n_experts=8, d_ff=None, dropout=0.1, top_k=3):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.n_experts = n_experts; self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.norm = nn.LayerNorm(d_model); self.load_loss = torch.tensor(0.)

    def forward(self, x):
        B, T, D = x.shape; flat = x.reshape(-1, D)
        logits = self.gate(flat); probs = F.softmax(logits, dim=-1)
        topk_v, topk_i = probs.topk(self.top_k, dim=-1)
        topk_v = topk_v / (topk_v.sum(-1, keepdim=True) + EPS)
        frac_t = probs.mean(0)
        frac_r = (topk_i == torch.arange(self.n_experts, device=x.device).unsqueeze(0)
                  ).float().mean(0)
        self.load_loss = self.n_experts * (frac_t * frac_r).sum()
        exp_out = torch.stack([exp(flat) for exp in self.experts], dim=1)
        idx_e   = topk_i.unsqueeze(-1).expand(-1, -1, D)
        out     = (exp_out.gather(1, idx_e) * topk_v.unsqueeze(-1)).sum(1)
        return self.norm(out.reshape(B, T, D) + x)


# ─────────────────────────────────────────────────────────────────────────────
# TFT v9 main model
# ─────────────────────────────────────────────────────────────────────────────
class TFTv9(nn.Module):
    def __init__(self, n_unk, n_kn, n_static, d_model=256, n_heads=8, n_lstm_layers=2,
                 dropout=0.12, enc_len=120, horizons=None, quantiles=None,
                 feat_group_ids=None, n_groups=11, n_experts=8, top_k_moe=3):
        super().__init__()
        self.d_model   = d_model; self.enc_len = enc_len
        self.horizons  = horizons or HORIZONS; self.n_h = len(self.horizons)
        self.max_h     = max(self.horizons)
        self.n_unk     = n_unk; self.n_kn = n_kn
        self.quantiles = quantiles or [0.1, 0.5, 0.9]; self.n_q = len(self.quantiles)

        # RevIN per-sample normalisation (encoder only)
        self.revin = RevIN(n_unk)

        # Feature embeddings
        self.unk_projs = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_unk)])
        self.kn_projs  = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_kn)])
        self.group_emb = nn.Embedding(n_groups + 1, d_model)
        self.feat_group_ids = feat_group_ids or [0] * n_unk

        # Static context
        self.static_proj  = nn.Linear(max(n_static, 1), d_model)
        self.static_grn_c = GRN(d_model, dropout=dropout)
        self.static_grn_h = GRN(d_model, dropout=dropout)
        self.static_grn_e = GRN(d_model, dropout=dropout)
        self.static_grn_d = GRN(d_model, dropout=dropout)
        self.regime_emb   = nn.Embedding(3, d_model)

        # Variable selection
        self.enc_vsn = VSN(n_unk + n_kn, d_model, d_ctx=d_model, dropout=dropout)
        self.dec_vsn = VSN(n_kn,         d_model, d_ctx=d_model, dropout=dropout)

        # Temporal processing
        self.tcn  = DilatedTCN(d_model, dropout=dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_lstm_layers,
                            batch_first=True,
                            dropout=dropout if n_lstm_layers > 1 else 0.)
        self.post_lstm_gate = GLU(d_model, d_model, dropout)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # Attention
        self.self_attn       = SelfAttnBlock(d_model, n_heads, dropout, max_len=512)
        self.cross_attn_day  = CrossAttnBlock(d_model, n_heads, dropout)
        self.cross_attn_week = CrossAttnBlock(d_model, max(n_heads // 2, 1), dropout)
        self.cross_attn_month = CrossAttnBlock(d_model, max(n_heads // 4, 1), dropout)
        self.scale_fuse      = nn.Linear(d_model * 3, d_model)
        self.post_attn_grn   = GRN(d_model, dropout=dropout)
        self.post_attn_norm  = nn.LayerNorm(d_model)

        # MoE FFN
        self.ff_moe = MoEFFN(d_model, n_experts=n_experts, d_ff=d_model * 4,
                             dropout=dropout, top_k=top_k_moe)

        # Regime gating
        self.regime_scale = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid())

        # Gate fusion (self-attn vs NTK-attn)
        self.gate_fusion = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

        # Prediction heads
        hh = d_model // 2
        self.point_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, hh), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hh, 1))
            for _ in self.horizons])

        self.quant_backbone = nn.Sequential(
            nn.Linear(d_model, hh), nn.GELU(), nn.Dropout(dropout))
        self.quant_heads = nn.ModuleList([
            nn.ModuleList([nn.Linear(hh, 1) for _ in self.quantiles])
            for _ in self.horizons])

        self.sign_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, hh // 2), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hh // 2, 1))
            for _ in self.horizons])

        self.vol_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, hh // 2), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hh // 2, 1), nn.Softplus())
            for _ in self.horizons])

        # Temperature for quantile calibration (learnable)
        self.temp = nn.Parameter(torch.ones(self.n_q))

        # Horizon embedding
        self.horizon_emb = nn.Embedding(self.n_h, d_model)
        self.input_drop  = nn.Dropout(dropout)

        # v9: temporal contrastive projection head
        self.contrast_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 64))

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _detect_regime(self, enc_unk):
        with torch.no_grad():
            std = enc_unk[..., 0].std(1)
            q33, q67 = std.quantile(0.33), std.quantile(0.67)
            r = torch.zeros(enc_unk.shape[0], dtype=torch.long, device=enc_unk.device)
            r[std > q33] = 1; r[std > q67] = 2
        return r

    def _embed_unk(self, x):
        gids  = torch.tensor(self.feat_group_ids, device=x.device, dtype=torch.long)
        g_bias = self.group_emb(gids)
        return [proj(x[..., i:i+1]) + g_bias[i]
                for i, proj in enumerate(self.unk_projs)]

    def _embed_kn(self, x):
        return [proj(x[..., i:i+1]) for i, proj in enumerate(self.kn_projs)]

    def _pool(self, x, stride):
        B, T, D = x.shape; Tn = T // stride
        if Tn == 0: return x.mean(1, keepdim=True)
        return x[:, :Tn * stride, :].reshape(B, Tn, stride, D).mean(2)

    def forward(self, enc_unk, enc_kn, dec_kn, static):
        B = enc_unk.shape[0]

        # RevIN on encoder window only
        enc_unk = self.revin(enc_unk, mode="norm")

        # Regime detection
        regime  = self._detect_regime(enc_unk)
        reg_emb = self.regime_emb(regime)

        # Static context
        static_in  = static if static.shape[-1] > 0 else torch.zeros(B, 1, device=enc_unk.device)
        static_emb = torch.tanh(self.static_proj(static_in)) + reg_emb
        c_s = self.static_grn_c(static_emb); h_s = self.static_grn_h(static_emb)
        c_e = self.static_grn_e(static_emb); c_d = self.static_grn_d(static_emb)
        h_init = h_s.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c_init = c_s.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()

        # Encoder VSN
        ctx_enc = c_e.unsqueeze(1).expand(-1, self.enc_len, -1)
        embs    = self._embed_unk(enc_unk) + self._embed_kn(enc_kn)
        if CFG.grad_ckpt and self.training:
            vsn_out, vsn_w = grad_ckpt(self.enc_vsn, embs, ctx_enc, use_reentrant=False)
        else:
            vsn_out, vsn_w = self.enc_vsn(embs, ctx_enc)
        vsn_out = self.input_drop(vsn_out)

        # TCN
        if CFG.grad_ckpt and self.training:
            tcn_out = grad_ckpt(self.tcn, vsn_out, use_reentrant=False)
        else:
            tcn_out = self.tcn(vsn_out)

        # Decoder VSN (calendar only — zero price future)
        dec_embs  = self._embed_kn(dec_kn)
        ctx_d     = c_d.unsqueeze(1).expand(-1, dec_kn.shape[1], -1)
        dec_feat, dec_vsn_w = self.dec_vsn(dec_embs, ctx_d)
        dec_feat  = self.input_drop(dec_feat)

        # LSTM over full sequence (encoder + decoder horizon)
        full_seq = torch.cat([tcn_out, dec_feat], dim=1)
        if CFG.grad_ckpt and self.training:
            lstm_out, _ = grad_ckpt(
                lambda x, hi, ci: self.lstm(x, (hi, ci))[0],
                full_seq, h_init, c_init, use_reentrant=False)
        else:
            lstm_out, _ = self.lstm(full_seq, (h_init, c_init))

        enc_lstm = lstm_out[:, :self.enc_len, :]
        dec_lstm = lstm_out[:, self.enc_len:, :]

        # Self-attention on encoder
        enc_self = self.self_attn(enc_lstm)
        # Gated fusion (no NTK attn in v9 to reduce params — use enc_self twice with gate)
        gate     = torch.sigmoid(self.gate_fusion(torch.cat([enc_self, enc_lstm], dim=-1)))
        enc_attn = gate * enc_self + (1 - gate) * enc_lstm

        # Post-LSTM gate on decoder
        gated = self.post_lstm_gate(dec_lstm)
        gated = self.post_lstm_norm(gated + dec_feat)

        # Multi-scale cross attention
        enc_week  = self._pool(enc_attn, 5)
        enc_month = self._pool(enc_attn, 21)
        ca_day    = self.cross_attn_day(gated,   enc_attn,  enc_attn)
        ca_week   = self.cross_attn_week(gated,  enc_week,  enc_week)
        ca_month  = self.cross_attn_month(gated, enc_month, enc_month)
        fused     = self.scale_fuse(torch.cat([ca_day, ca_week, ca_month], dim=-1))
        fused     = self.post_attn_grn(fused)
        fused     = self.post_attn_norm(fused + gated)

        # MoE FFN
        ff_out   = self.ff_moe(fused)
        r_scale  = self.regime_scale(reg_emb).unsqueeze(1)
        ff_out   = ff_out * (0.5 + r_scale)

        # v9: contrastive projection (encoder CLS token = mean)
        enc_cls  = enc_attn.mean(1)
        contrast_emb = self.contrast_proj(enc_cls)  # [B, 64]

        T_dec = ff_out.shape[1]; temp = F.softplus(self.temp)
        all_points = []; all_quants = []; all_signs = []; all_vols = []

        for hi, h in enumerate(self.horizons):
            t_idx = min(h - 1, T_dec - 1)
            h_emb = self.horizon_emb(torch.tensor(hi, device=ff_out.device))
            rep   = ff_out[:, t_idx, :] + h_emb
            point = self.point_heads[hi](rep).squeeze(-1)
            qbase = self.quant_backbone(rep)
            raw_q = torch.stack([qh(qbase) for qh in self.quant_heads[hi]], dim=-1).squeeze(1)
            # Monotone sort: ensures q25 ≤ q50 ≤ q75 (no crossing)
            raw_q_sorted, _ = raw_q.sort(dim=-1)
            quants = raw_q_sorted / temp
            sign   = self.sign_heads[hi](rep).squeeze(-1)
            vol    = self.vol_heads[hi](rep).squeeze(-1)
            all_points.append(point); all_quants.append(quants)
            all_signs.append(sign);   all_vols.append(vol)

        points = torch.stack(all_points, dim=1)
        quants = torch.stack(all_quants, dim=1)
        signs  = torch.stack(all_signs,  dim=1)
        vols   = torch.stack(all_vols,   dim=1)

        # Store for visualisation
        self._enc_vsn_w    = vsn_w
        self._dec_vsn_w    = dec_vsn_w
        self._attn_w       = self.cross_attn_day._attn_w
        self._moe_load     = self.ff_moe.load_loss
        self._contrast_emb = contrast_emb

        return points, quants, signs, vols


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def quantile_loss(pred_q, target, quantiles, smooth=0.03):
    losses = []
    for qi, q in enumerate(quantiles):
        qs  = q * (1 - smooth) + 0.5 * smooth
        err = target - pred_q[..., qi]
        losses.append(torch.max(qs * err, (qs - 1) * err))
    return torch.stack(losses, -1).mean()

def asymmetric_huber(pred, target, delta=0.008, alpha=1.6):
    err = target - pred; ab = err.abs()
    h   = torch.where(ab < delta, 0.5 * err ** 2 / delta, ab - 0.5 * delta)
    return torch.where(err > 0, alpha * h, h).mean()

def focal_bce(sign_logit, target, gamma=2.0):
    st = (target > 0).float()
    p  = torch.sigmoid(sign_logit)
    pt = torch.where(st > 0.5, p, 1. - p).clamp(EPS, 1. - EPS)
    return (-(1 - pt) ** gamma * torch.log(pt)).mean()

def soft_ic_loss(pred, target):
    def _norm(x):
        mu = x.mean(-1, keepdim=True); sigma = x.std(-1, keepdim=True) + EPS
        return (x - mu) / sigma
    return -((_norm(pred) * _norm(target)).mean())

def ndcg_surrogate_loss(pred, target, k=20):
    if pred.dim() == 1: pred = pred.unsqueeze(0); target = target.unsqueeze(0)
    B, H = pred.shape; loss = torch.tensor(0., device=pred.device, requires_grad=True)
    for h in range(H):
        p = pred[:, h]; t = target[:, h]
        diff_p = p.unsqueeze(1) - p.unsqueeze(0)
        diff_t = t.unsqueeze(1) - t.unsqueeze(0)
        label  = (diff_t > 0).float()
        rank_t = t.argsort(descending=True).argsort().float()
        idcg   = (1. / torch.log2(rank_t + 2.)).sum()
        weight = (1. / (torch.log2(rank_t.unsqueeze(1) + 2.) +
                         torch.log2(rank_t.unsqueeze(0) + 2.))).abs()
        loss = loss + (weight * F.binary_cross_entropy_with_logits(
            diff_p, label, reduction="none")).mean() / (idcg.clamp(min=EPS) * H)
    return loss / B

def sharpe_loss(pred, vol, target):
    pred_sharpe = pred / (vol + EPS)
    real_sharpe = target / (target.std(dim=-1, keepdim=True) + EPS)
    return F.mse_loss(pred_sharpe, real_sharpe.detach())

def temporal_contrastive_loss(emb, temperature=0.1):
    """
    Positive pairs: consecutive time steps from the same ticker (same batch).
    Negative pairs: all other samples in the batch.
    Uses NT-Xent (SimCLR-style).
    """
    B = emb.shape[0]
    if B < 4: return torch.tensor(0., device=emb.device)
    # Normalise
    emb = F.normalize(emb, dim=-1)
    sim = emb @ emb.T / temperature  # [B, B]
    # Positives: sequential pairs (i, i+1) if B is even; use alternating halves
    half = B // 2
    z1 = emb[:half]; z2 = emb[half:half * 2]
    sim12 = (z1 * z2).sum(-1) / temperature
    sim_all = (z1 @ emb.T) / temperature
    loss = -sim12 + torch.logsumexp(sim_all, dim=-1)
    return loss.mean()

def combined_loss(points, quants, signs, vols, targets, quantiles,
                  lambda_q, lambda_sign, lambda_ic, lambda_ndcg, lambda_sharpe,
                  label_smooth, moe_load, contrast_emb=None, lambda_contrast=0.0):
    huber  = asymmetric_huber(points, targets)
    ql     = quantile_loss(quants, targets, quantiles, label_smooth)
    bce    = focal_bce(signs, targets)
    ic_l   = soft_ic_loss(points, targets)
    ndcg_l = ndcg_surrogate_loss(points, targets)
    sharpe_l = sharpe_loss(points, vols, targets)

    lam_base = 1. - lambda_q - lambda_sign - lambda_ic - lambda_ndcg - lambda_sharpe
    total = (lam_base * huber + lambda_q * ql + lambda_sign * bce +
             lambda_ic * ic_l + lambda_ndcg * ndcg_l + lambda_sharpe * sharpe_l)

    if moe_load is not None and isinstance(moe_load, torch.Tensor):
        total = total + 1e-3 * moe_load

    if contrast_emb is not None and lambda_contrast > 0:
        total = total + lambda_contrast * temporal_contrastive_loss(contrast_emb)

    return total, huber, ql, bce, ic_l


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred, target, quantiles, quant_preds=None, horizon_names=None):
    pred = np.asarray(pred); target = np.asarray(target)
    if pred.ndim == 1: pred = pred[:, None]; target = target[:, None]
    N, H = pred.shape; results = {}
    for h in range(H):
        p = pred[:, h]; t = target[:, h]
        mse  = float(np.mean((p - t) ** 2))
        mae  = float(np.mean(np.abs(p - t)))
        ic   = float(np.corrcoef(p, t)[0, 1]) if N > 1 else 0.
        ric, _ = scipy_stats.spearmanr(p, t)
        da   = float(np.mean(np.sign(p) == np.sign(t)))
        med  = np.median(p); lr = np.where(p > med, t, -t)
        sharpe  = (lr.mean() / (lr.std() + EPS)) * math.sqrt(252)
        cum = np.cumprod(1 + lr.clip(-0.5, 0.5))
        rm  = np.maximum.accumulate(cum)
        dd  = ((rm - cum) / (rm + EPS)).max()
        calmar = (lr.mean() * 252) / (dd + EPS)
        q75 = np.percentile(p, 75); q25 = np.percentile(p, 25)
        q4_ret = t[p >= q75].mean() if (p >= q75).any() else 0.
        q1_ret = t[p <= q25].mean() if (p <= q25).any() else 0.
        suffix = f"_h{horizon_names[h]}" if horizon_names else f"_h{h}"
        results.update({
            f"mse{suffix}": mse, f"mae{suffix}": mae,
            f"IC{suffix}": ic, f"RankIC{suffix}": float(ric),
            f"DA{suffix}": da, f"Sharpe{suffix}": sharpe,
            f"Calmar{suffix}": calmar, f"Q4Q1{suffix}": q4_ret - q1_ret,
        })
    results["IC_mean"]     = float(np.mean([v for k, v in results.items() if "IC_h"     in k and "Rank" not in k]))
    results["RankIC_mean"] = float(np.mean([v for k, v in results.items() if "RankIC"   in k]))
    results["DA_mean"]     = float(np.mean([v for k, v in results.items() if "DA_h"     in k]))
    results["Sharpe_mean"] = float(np.mean([v for k, v in results.items() if "Sharpe_h" in k]))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosine:
    def __init__(self, opt, warmup, total, min_lr):
        self.opt = opt; self.warmup = warmup; self.total = total; self.min_lr = min_lr
        self.base_lr = opt.param_groups[0]["lr"]; self.epoch = 0

    def step(self):
        e = self.epoch
        if e < self.warmup:
            lr = self.base_lr * (e + 1) / self.warmup
        else:
            prog = (e - self.warmup) / max(self.total - self.warmup, 1)
            lr   = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * prog))
        for pg in self.opt.param_groups: pg["lr"] = lr
        self.epoch += 1; return lr


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class TFTTrainer:
    def __init__(self, model, train_dl, val_dl, test_dl, norm_stats,
                 unk_names, kn_names, quantiles, horizons, device, use_amp, n_gpus=1):
        self.device    = device; self.use_amp = use_amp; self.n_gpus = n_gpus
        self.quantiles = quantiles; self.horizons = horizons; self.n_h = len(horizons)
        self.unk_names = unk_names; self.kn_names = kn_names; self.norm_stats = norm_stats
        self.train_dl  = train_dl; self.val_dl = val_dl; self.test_dl = test_dl

        if n_gpus > 1:
            ts(f"Wrapping in DataParallel x{n_gpus}")
            model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
        self.model = model.to(device)

        base_params = list(self.model.parameters())
        if CFG.use_sam:
            self.opt = SAM(base_params, torch.optim.AdamW, rho=CFG.sam_rho,
                           lr=CFG.lr, weight_decay=CFG.weight_decay)
            sched_opt = self.opt.base_optimizer
        else:
            self.opt = torch.optim.AdamW(base_params, lr=CFG.lr, weight_decay=CFG.weight_decay)
            sched_opt = self.opt
        self.sched  = WarmupCosine(sched_opt, CFG.warmup_epochs, CFG.epochs, CFG.min_lr)
        self.scaler = GradScaler() if use_amp else None

        raw = self._raw_model()
        self.swa_model  = torch.optim.swa_utils.AveragedModel(raw)
        self.swa_active = False; self.swa_start = int(CFG.epochs * CFG.swa_start_frac)
        self.best_state    = None; self.best_val_loss = float("inf"); self.patience_cnt = 0
        # EMA of val loss for smoother patience decisions
        self._val_loss_ema = None
        self.history = {k: [] for k in [
            "epoch","train_loss","val_loss","val_ic_mean","val_rank_ic_mean",
            "val_da_mean","val_sharpe_mean","lr","grad_norm"]}
        self._train_start = time.time()

    def _raw_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _to(self, batch):
        eu, ek, dk, tgt, sv = batch
        return (eu.to(self.device, non_blocking=True),
                ek.to(self.device, non_blocking=True),
                dk.to(self.device, non_blocking=True),
                tgt.to(self.device, non_blocking=True),
                sv.to(self.device, non_blocking=True))

    def _fwd(self, eu, ek, dk, tgt, sv):
        pts, qts, signs, vols = self.model(eu, ek, dk, sv)
        raw = self._raw_model()
        moe_l      = getattr(raw, "_moe_load", None)
        contrast_e = getattr(raw, "_contrast_emb", None)
        loss, h, q, b, ic = combined_loss(
            pts, qts, signs, vols, tgt, self.quantiles,
            CFG.lambda_q, CFG.lambda_sign, CFG.lambda_ic, CFG.lambda_ndcg,
            CFG.lambda_sharpe, CFG.label_smooth, moe_l,
            contrast_emb=contrast_e, lambda_contrast=CFG.lambda_contrastive)
        return loss, pts, qts

    def train_epoch(self, epoch):
        self.model.train(); total_loss = 0.; n = 0; total_gn = 0.; nb = len(self.train_dl)
        self.opt.zero_grad(set_to_none=True)

        for bi, batch in enumerate(self.train_dl):
            eu, ek, dk, tgt, sv = self._to(batch)
            is_acc = ((bi + 1) % CFG.grad_accum == 0) or (bi == nb - 1)

            if CFG.use_sam:
                if self.use_amp:
                    with autocast(): loss, *_ = self._fwd(eu, ek, dk, tgt, sv)
                    self.scaler.scale(loss / CFG.grad_accum).backward()
                    if is_acc:
                        self.scaler.unscale_(self.opt.base_optimizer)
                        gn = nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
                        self.opt.first_step(zero_grad=True)
                        with autocast(): loss2, *_ = self._fwd(eu, ek, dk, tgt, sv)
                        self.scaler.scale(loss2).backward()
                        self.scaler.unscale_(self.opt.base_optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
                        self.opt.second_step(zero_grad=True); self.scaler.update()
                else:
                    loss, *_ = self._fwd(eu, ek, dk, tgt, sv)
                    (loss / CFG.grad_accum).backward()
                    if is_acc:
                        gn = nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
                        self.opt.first_step(zero_grad=True)
                        loss2, *_ = self._fwd(eu, ek, dk, tgt, sv); loss2.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
                        self.opt.second_step(zero_grad=True)
            else:
                if self.use_amp:
                    with autocast(): loss, *_ = self._fwd(eu, ek, dk, tgt, sv)
                    self.scaler.scale(loss / CFG.grad_accum).backward()
                    if is_acc:
                        self.scaler.unscale_(self.opt)
                        gn = nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
                        self.scaler.step(self.opt); self.scaler.update()
                        self.opt.zero_grad(set_to_none=True)
                else:
                    loss, *_ = self._fwd(eu, ek, dk, tgt, sv)
                    (loss / CFG.grad_accum).backward()
                    if is_acc:
                        gn = nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
                        self.opt.step(); self.opt.zero_grad(set_to_none=True)

            total_loss += loss.item() * eu.shape[0]; n += eu.shape[0]
            if is_acc: total_gn += float(gn) if isinstance(gn, torch.Tensor) else float(gn)

            if (bi + 1) % max(1, nb // 5) == 0:
                done_frac = (epoch - 1 + bi / nb) / CFG.epochs
                ram_gb = _check_ram_gb()
                print(f"  E{epoch} [{bi+1}/{nb}] loss={total_loss/max(n,1):.5f} | "
                      f"RAM={ram_gb:.1f}GB | ETA={eta(done_frac, self._train_start)}",
                      flush=True)

        if epoch >= self.swa_start:
            if not self.swa_active: ts(f"SWA activated at epoch {epoch}"); self.swa_active = True
            self.swa_model.update_parameters(self._raw_model())

        return total_loss / max(n, 1), total_gn / max(nb, 1)

    @torch.no_grad()
    def eval_epoch(self, dl, tta=False):
        m = self.swa_model if self.swa_active else self._raw_model()
        m.train() if tta else m.eval()
        total_loss = 0.; n = 0; all_pts = []; all_qts = []; all_tgt = []
        for batch in dl:
            eu, ek, dk, tgt, sv = self._to(batch)
            if tta:
                plist = []; qlist = []
                for _ in range(CFG.tta_passes):
                    pt, qt, si, vl = m(eu, ek, dk, sv)
                    plist.append(pt); qlist.append(qt)
                pts = torch.stack(plist).mean(0); qts = torch.stack(qlist).mean(0)
                _, _, si, vl = m(eu, ek, dk, sv)
            else:
                pts, qts, si, vl = m(eu, ek, dk, sv)
            moe_l = getattr(m, "_moe_load", None)
            loss, *_ = combined_loss(pts, qts, si, vl, tgt, self.quantiles,
                                     CFG.lambda_q, CFG.lambda_sign, CFG.lambda_ic,
                                     CFG.lambda_ndcg, CFG.lambda_sharpe, CFG.label_smooth, moe_l)
            total_loss += loss.item() * eu.shape[0]; n += eu.shape[0]
            all_pts.append(pts.cpu()); all_qts.append(qts.cpu()); all_tgt.append(tgt.cpu())
        ap = torch.cat(all_pts); aq = torch.cat(all_qts); at = torch.cat(all_tgt)
        metrics = compute_metrics(ap.numpy(), at.numpy(), self.quantiles, aq.numpy(),
                                  horizon_names=[str(h) for h in self.horizons])
        return total_loss / max(n, 1), metrics, ap, aq, at

    def train(self):
        ts(f"Training: {CFG.epochs} epochs | {self.n_gpus} GPUs | "
           f"SAM={CFG.use_sam} | AMP={self.use_amp} | GradCkpt={CFG.grad_ckpt} | "
           f"GradAccum={CFG.grad_accum} | PurgeGap={CFG.purge_gap}")
        for epoch in range(1, CFG.epochs + 1):
            t0 = time.time()
            ts(f"Epoch {epoch}/{CFG.epochs} — training...")
            tl, gn = self.train_epoch(epoch)
            lr = self.sched.step()
            ts(f"Epoch {epoch}/{CFG.epochs} — validating...")
            vl, vm, _, _, _ = self.eval_epoch(self.val_dl)
            elapsed = time.time() - t0

            # EMA val loss for stable patience
            alpha_ema = 0.3
            if self._val_loss_ema is None: self._val_loss_ema = vl
            self._val_loss_ema = alpha_ema * vl + (1 - alpha_ema) * self._val_loss_ema

            ts(f"E{epoch:03d} {elapsed:.0f}s | trn={tl:.5f} val={vl:.5f} "
               f"(ema={self._val_loss_ema:.5f}) | IC={vm['IC_mean']:.4f} "
               f"RIC={vm['RankIC_mean']:.4f} DA={vm['DA_mean']:.3f} "
               f"Sharpe={vm['Sharpe_mean']:.3f} | lr={lr:.2e} gn={gn:.3f} "
               f"pat={self.patience_cnt}/{CFG.patience} | "
               f"TotalETA={eta(epoch/CFG.epochs, self._train_start)}")

            for k, v in [("epoch", epoch), ("train_loss", tl), ("val_loss", vl),
                         ("val_ic_mean", vm["IC_mean"]), ("val_rank_ic_mean", vm["RankIC_mean"]),
                         ("val_da_mean", vm["DA_mean"]), ("val_sharpe_mean", vm["Sharpe_mean"]),
                         ("lr", lr), ("grad_norm", gn)]:
                self.history[k].append(v)

            if self._val_loss_ema < self.best_val_loss:
                self.best_val_loss = self._val_loss_ema
                raw = self._raw_model()
                self.best_state = {k: v.clone() for k, v in raw.state_dict().items()}
                self.patience_cnt = 0
                torch.save(self.best_state, os.path.join(CFG.out_dir, "best_model.pt"))
                ts(f"  New best val_loss={self._val_loss_ema:.5f} — model saved")
            else:
                self.patience_cnt += 1
                if self.patience_cnt >= CFG.patience:
                    ts(f"Early stop at epoch {epoch}"); break

        if self.best_state:
            self._raw_model().load_state_dict(self.best_state)
            ts(f"Best model restored — val_loss={self.best_val_loss:.5f}")

        if self.swa_active:
            ts("Updating SWA batch-norm statistics...")
            try:
                torch.optim.swa_utils.update_bn(self.train_dl, self.swa_model, device=self.device)
            except Exception as e:
                ts(f"SWA BN skipped: {e}")

    @torch.no_grad()
    def collect_vsn_weights(self, dl, nb=15):
        m = self._raw_model(); m.eval(); ws = []
        for i, batch in enumerate(dl):
            if i >= nb: break
            eu, ek, dk, _, sv = self._to(batch); m(eu, ek, dk, sv)
            w = getattr(m, "_enc_vsn_w", None)
            if w is not None: ws.append(w.cpu().numpy())
        return np.mean(np.concatenate(ws, 0), (0, 1)) if ws else np.zeros(self.n_h)

    @torch.no_grad()
    def collect_attn(self, dl, nb=5):
        m = self._raw_model(); m.eval(); ws = []
        for i, batch in enumerate(dl):
            if i >= nb: break
            eu, ek, dk, _, sv = self._to(batch); m(eu, ek, dk, sv)
            w = getattr(m, "_attn_w", None)
            if w is not None: ws.append(w.mean(1).cpu().numpy())
        return np.concatenate(ws, 0).mean(0) if ws else None


# ─────────────────────────────────────────────────────────────────────────────
# Visualiser — 20 charts
# ─────────────────────────────────────────────────────────────────────────────
class TFTVisualizer:
    def __init__(self, trainer, out_dir):
        self.t = trainer; self.out = out_dir; self.h = trainer.history
        self.hn = [str(h) for h in trainer.horizons]
        self._tp = self._tq = self._tt = self._tm = None

    def _save(self, fig, name):
        path = os.path.join(self.out, f"{name}.html")
        pio.write_html(fig, path, include_plotlyjs="cdn")
        ts(f"Chart saved: {name}.html")

    def _get_test(self):
        if self._tp is None:
            ts("Running TTA test evaluation for charts...")
            _, m, p, q, t = self.t.eval_epoch(self.t.test_dl, tta=True)
            self._tp = p.numpy(); self._tq = q.numpy()
            self._tt = t.numpy(); self._tm = m
        return self._tp, self._tq, self._tt, self._tm

    def c01_training_curves(self):
        h = self.h; fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
            subplot_titles=["Loss", "IC Mean", "DA Mean", "Gradient Norm"])
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_loss"], name="Train",
                                 line=dict(color="#4E79A7", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_loss"],   name="Val",
                                 line=dict(color="#F28E2B", width=2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_ic_mean"], name="IC",
                                 line=dict(color="#59A14F")), row=2, col=1)
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_da_mean"], name="DA",
                                 line=dict(color="#B07AA1")), row=3, col=1)
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["grad_norm"],   name="GN",
                                 line=dict(color="#E15759")), row=4, col=1)
        fig.update_layout(title="TFT v9 Training Dynamics", template=DARK, height=900)
        self._save(fig, "01_training_curves")

    def c02_val_metrics(self):
        h = self.h; fig = make_subplots(rows=2, cols=2,
            subplot_titles=["IC", "Rank IC", "DA", "Sharpe"])
        combos = [("val_ic_mean","#59A14F","IC"),("val_rank_ic_mean","#4E79A7","Rank IC"),
                  ("val_da_mean","#E15759","DA"),("val_sharpe_mean","#F28E2B","Sharpe")]
        for i, (k, c, t2) in enumerate(combos):
            r, col = divmod(i, 2)
            fig.add_trace(go.Scatter(x=h["epoch"], y=h[k], name=t2,
                                     line=dict(color=c, width=2)), row=r+1, col=col+1)
        fig.update_layout(title="Validation Metrics", template=DARK,
                          height=550, showlegend=False)
        self._save(fig, "02_val_metrics")

    def c03_vsn_importance(self):
        ts("Computing VSN feature importance...")
        w = self.t.collect_vsn_weights(self.t.val_dl)
        names = self.t.unk_names + self.t.kn_names
        n = min(len(w), len(names)); w = w[:n]; names = names[:n]
        si = np.argsort(w)[::-1][:50]
        fig = go.Figure(go.Bar(
            x=[float(w[i]) for i in si], y=[names[i] for i in si],
            orientation="h",
            marker=dict(color=[float(w[i]) for i in si], colorscale="Viridis")))
        fig.update_layout(title="Top-50 Feature Importances (VSN)",
                          yaxis=dict(autorange="reversed"), template=DARK, height=1100)
        self._save(fig, "03_vsn_importance")

    def c04_attn_heatmap(self):
        attn = self.t.collect_attn(self.t.val_dl)
        if attn is None: return
        Tq, Tk = attn.shape
        fig = go.Figure(go.Heatmap(z=attn, x=list(range(Tk)),
                                   y=[f"q{i}" for i in range(Tq)], colorscale="Plasma"))
        fig.update_layout(title="Cross-Attention (ALiBi): Decoder→Encoder",
                          template=DARK, height=400)
        self._save(fig, "04_attn_heatmap")

    def c05_pred_vs_actual(self):
        tp, tq, tt, _ = self._get_test()
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=[f"{d}d" for d in self.t.horizons])
        for hi, h2 in enumerate(self.t.horizons):
            p = tp[:, hi][:3000]; t2 = tt[:, hi][:3000]
            ic = float(np.corrcoef(p, t2)[0, 1]) if len(p) > 2 else 0.
            r, c = divmod(hi, 3)
            fig.add_trace(go.Scatter(x=t2, y=p, mode="markers",
                                     marker=dict(size=3, color="#4E79A7", opacity=0.4),
                                     name=f"{h2}d IC={ic:.3f}"), row=r+1, col=c+1)
        fig.update_layout(title="Predicted vs Actual by Horizon (TTA)",
                          template=DARK, height=600)
        self._save(fig, "05_pred_vs_actual")

    def c06_ic_by_horizon(self):
        _, _, _, m = self._get_test()
        ics  = [m.get(f"IC_h{h}", 0.) for h in self.t.horizons]
        rics = [m.get(f"RankIC_h{h}", 0.) for h in self.t.horizons]
        labels = [f"{h}d" for h in self.t.horizons]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=ics, name="Pearson IC",
                             marker=dict(color=ics, colorscale="RdYlGn", cmin=-0.05, cmax=0.12)))
        fig.add_trace(go.Scatter(x=labels, y=rics, mode="lines+markers", name="Rank IC",
                                 line=dict(color="#F28E2B", width=2)))
        fig.add_hline(y=0, line_dash="dot", line_color="white")
        fig.update_layout(title="IC by Prediction Horizon", template=DARK, height=420)
        self._save(fig, "06_ic_by_horizon")

    def c07_quantile_fan(self):
        tp, tq, tt, _ = self._get_test(); h_idx = 0
        p = tp[:, h_idx][:300]; t2 = tt[:, h_idx][:300]; x = list(range(len(p)))
        nq = len(self.t.quantiles); fig = go.Figure()
        for (li, hi2), alpha in [((0, nq-1), 0.07), ((1, nq-2), 0.13), ((2, nq-3), 0.21)]:
            if hi2 <= li: continue
            lo_q = tq[:, h_idx, li][:len(x)]; hi_q = tq[:, h_idx, hi2][:len(x)]
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=np.concatenate([hi_q, lo_q[::-1]]).tolist(),
                fill="toself", fillcolor=f"rgba(78,121,167,{alpha})", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=x, y=t2.tolist(), mode="lines",
                                 line=dict(color="white", width=1.5), name="Actual"))
        fig.add_trace(go.Scatter(x=x, y=p.tolist(),  mode="lines",
                                 line=dict(color="#F28E2B", width=1.5, dash="dash"), name="Pred"))
        fig.update_layout(title="Quantile Fan (1d horizon)", template=DARK, height=480)
        self._save(fig, "07_quantile_fan")

    def c08_residual_dist(self):
        tp, _, tt, _ = self._get_test()
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=[f"{h}d" for h in self.t.horizons])
        for hi, h2 in enumerate(self.t.horizons):
            res = tp[:, hi] - tt[:, hi]
            sk = float(scipy_stats.skew(res)); ku = float(scipy_stats.kurtosis(res))
            r, c = divmod(hi, 3)
            fig.add_trace(go.Histogram(x=res, nbinsx=80, marker_color="#4E79A7",
                                       name=f"{h2}d sk={sk:.2f} ku={ku:.2f}"), row=r+1, col=c+1)
        fig.update_layout(title="Residual Distributions by Horizon",
                          template=DARK, height=600, showlegend=False)
        self._save(fig, "08_residual_dist")

    def c09_feature_corr_matrix(self):
        top30 = self.t.unk_names[:30]; rows = []
        for i, batch in enumerate(self.t.val_dl):
            if i >= 10: break
            rows.append(batch[0].numpy().reshape(-1, batch[0].shape[-1]))
        if not rows: return
        flat = np.concatenate(rows)
        idx  = [self.t.unk_names.index(n) for n in top30 if n in self.t.unk_names][:30]
        if len(idx) < 2: return
        corr = np.corrcoef(flat[:, idx].T); nms = [self.t.unk_names[i] for i in idx]
        fig  = go.Figure(go.Heatmap(z=np.round(corr, 3), x=nms, y=nms,
                                    colorscale="RdBu", zmin=-1, zmax=1))
        fig.update_layout(title="Feature Correlation Matrix (Top-30)",
                          template=DARK, height=800, xaxis_tickangle=-45)
        self._save(fig, "09_feature_corr_matrix")

    def c10_ls_backtest(self):
        tp, _, tt, _ = self._get_test(); fig = go.Figure()
        colors = ["#59A14F","#4E79A7","#F28E2B","#E15759","#B07AA1","#76B7B2"]
        for hi, h2 in enumerate(self.t.horizons):
            p = tp[:, hi]; t2 = tt[:, hi]; med = np.median(p)
            lr  = np.where(p > med, t2, -t2)
            cum = np.cumprod(1 + lr.clip(-0.5, 0.5)) - 1
            fig.add_trace(go.Scatter(y=cum, name=f"{h2}d",
                                     line=dict(color=colors[hi], width=1.8)))
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(title="L/S Portfolio Backtest by Horizon",
                          template=DARK, height=480)
        self._save(fig, "10_ls_backtest")

    def c11_rolling_ic(self):
        tp, _, tt, _ = self._get_test()
        window = min(500, len(tp) // 5)
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=[f"Rolling IC {h}d" for h in self.t.horizons])
        for hi, h2 in enumerate(self.t.horizons):
            x_vals = []; rics = []
            for i in range(window, len(tp), max(1, window // 2)):
                p = tp[i - window:i, hi]; t2 = tt[i - window:i, hi]
                rics.append(float(np.corrcoef(p, t2)[0, 1]) if len(p) > 2 else 0.)
                x_vals.append(i)
            r, c = divmod(hi, 3)
            fig.add_trace(go.Scatter(x=x_vals, y=rics, line=dict(width=1.5), name=f"{h2}d"),
                          row=r+1, col=c+1)
            fig.add_hline(y=0, line_dash="dot", line_color="grey", row=r+1, col=c+1)
        fig.update_layout(title="Rolling IC by Horizon", template=DARK,
                          height=600, showlegend=False)
        self._save(fig, "11_rolling_ic")

    def c12_quantile_calibration(self):
        _, tq, tt, _ = self._get_test(); nom = self.t.quantiles
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=[f"{h}d" for h in self.t.horizons])
        for hi, h2 in enumerate(self.t.horizons):
            obs = [float(np.mean(tt[:, hi] < tq[:, hi, qi])) for qi in range(len(nom))]
            r, c = divmod(hi, 3)
            fig.add_trace(go.Scatter(x=nom, y=obs, mode="lines+markers",
                                     line=dict(color="#F28E2B", width=2)), row=r+1, col=c+1)
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="white", dash="dot")), row=r+1, col=c+1)
        fig.update_layout(title="Quantile Calibration by Horizon",
                          template=DARK, height=600, showlegend=False)
        self._save(fig, "12_quantile_calibration")

    def c13_decile_returns(self):
        tp, _, tt, _ = self._get_test(); h_idx = 0
        p = tp[:, h_idx]; t2 = tt[:, h_idx]
        dec = np.percentile(p, np.arange(10, 100, 10))
        bins = np.digitize(p, dec)
        mr = []; sr = []; labels = []
        for d in range(10):
            mask = (bins == d)
            if mask.sum() > 5:
                mr.append(t2[mask].mean()); sr.append(t2[mask].std()); labels.append(f"D{d+1}")
        colors = ["#E15759" if r < 0 else "#59A14F" for r in mr]
        fig = go.Figure(go.Bar(x=labels, y=mr,
                                error_y=dict(type="data", array=sr, visible=True),
                                marker_color=colors))
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(title="Mean Return by Signal Decile (1d)",
                          template=DARK, height=430)
        self._save(fig, "13_decile_returns")

    def c14_regime_ic(self):
        m2 = self.t._raw_model(); m2.eval(); tp = []; tt = []; regimes = []
        with torch.no_grad():
            for i, batch in enumerate(self.t.test_dl):
                if i >= 30: break
                eu, ek, dk, tgt, sv = self.t._to(batch)
                pt, _, _, _ = m2(eu, ek, dk, sv)
                std = eu[..., 0].std(1); q33, q67 = std.quantile(0.33), std.quantile(0.67)
                r   = torch.zeros(eu.shape[0], dtype=torch.long)
                r[std.cpu() > q33.cpu()] = 1; r[std.cpu() > q67.cpu()] = 2
                tp.append(pt.cpu().numpy()); tt.append(tgt.cpu().numpy())
                regimes.append(r.numpy())
        if not tp: return
        tp = np.concatenate(tp); tt = np.concatenate(tt); regimes = np.concatenate(regimes)
        names = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}
        ics = []; das = []; labels = []
        for r2, name in names.items():
            mask = (regimes == r2)
            if mask.sum() < 10: continue
            ic = float(np.corrcoef(tp[mask, 0], tt[mask, 0])[0, 1])
            da = float(np.mean(np.sign(tp[mask, 0]) == np.sign(tt[mask, 0])))
            ics.append(ic); das.append(da); labels.append(name)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["IC by Regime", "DA by Regime"])
        colors = ["#4E79A7", "#59A14F", "#E15759"][:len(labels)]
        fig.add_trace(go.Bar(x=labels, y=ics, marker_color=colors), row=1, col=1)
        fig.add_trace(go.Bar(x=labels, y=das, marker_color=colors), row=1, col=2)
        fig.add_hline(y=0, line_dash="dot", row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dot", row=1, col=2)
        fig.update_layout(title="Performance by Volatility Regime",
                          template=DARK, height=420, showlegend=False)
        self._save(fig, "14_regime_ic")

    def c15_horizon_sharpe(self):
        tp, _, tt, _ = self._get_test(); sharpes = []; labels = []
        for hi, h2 in enumerate(self.t.horizons):
            p = tp[:, hi]; t2 = tt[:, hi]; med = np.median(p)
            lr = np.where(p > med, t2, -t2)
            s  = (lr.mean() / (lr.std() + EPS)) * math.sqrt(252)
            sharpes.append(s); labels.append(f"{h2}d")
        colors = ["#59A14F" if s > 0 else "#E15759" for s in sharpes]
        fig = go.Figure(go.Bar(x=labels, y=sharpes, marker_color=colors))
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(title="Annualised L/S Sharpe by Horizon",
                          template=DARK, height=400)
        self._save(fig, "15_horizon_sharpe")

    def c16_tta_uncertainty(self):
        m2 = self.t._raw_model(); m2.train(); vs = []; ts_ = []
        with torch.no_grad():
            for i, batch in enumerate(self.t.test_dl):
                if i >= 20: break
                eu, ek, dk, tgt, sv = self.t._to(batch)
                plist = [m2(eu, ek, dk, sv)[0] for _ in range(CFG.tta_passes)]
                var = torch.stack(plist).var(0).mean(1)
                vs.append(var.cpu().numpy()); ts_.append(tgt[:, 0].abs().cpu().numpy())
        m2.eval()
        if not vs: return
        vv = np.concatenate(vs); tv = np.concatenate(ts_)
        fig = go.Figure(go.Scatter(x=tv, y=vv, mode="markers",
                                   marker=dict(size=3, color="#4E79A7", opacity=0.5)))
        fig.update_layout(title="TTA Prediction Variance vs |Actual Return|",
                          template=DARK, height=420)
        self._save(fig, "16_tta_uncertainty")

    def c17_group_importance(self):
        w  = self.t.collect_vsn_weights(self.t.val_dl)
        uw = w[:len(self.t.unk_names)]
        GROUPS = {
            "momentum":["mom_","return_","macd","pmo","lr_slope"],
            "volatility":["realized_vol","vol_","garman","parkinson","atr","fractal","spectral","var_ratio"],
            "trend":["ema_ratio","adx","trend_","hurst","close_to_ema","price_to_ma","kalman"],
            "mean_rev":["rsi","stoch","williams","omega","rolling_entropy","entropy_"],
            "volume":["obv","volume_","dollar_flow","amihud","kyle","liq_","mfi","chaikin","vpt","vwap","orderflow","flow_toxicity","vwm"],
            "risk":["sharpe","sortino","calmar","tail","drawdown","residual","omega_ratio"],
            "price_act":["body","hl_spread","gap","breakout","wick","intraday","oc_return"],
            "market":["market_","spy_","breadth","regime","sector","cov_spy_"],
            "calendar":["day_of_week","month_","week_","quarter"],
            "microstructure":["microstructure","informed_trading","price_impact","beta_decay","realized_skew"],
            "other":[],
        }
        gw = {g: 0. for g in GROUPS}
        for name, ww in zip(self.t.unk_names, uw):
            a = "other"
            for gn, ps in GROUPS.items():
                if ps and any(name.startswith(p) or p in name for p in ps): a = gn; break
            gw[a] += float(ww)
        gs = list(gw.keys()); vs = [gw[g] for g in gs]
        fig = go.Figure(go.Pie(labels=gs, values=vs, hole=0.4,
                                marker=dict(colors=px.colors.qualitative.Plotly)))
        fig.update_layout(title="Feature Group Importance (VSN)",
                          template=DARK, height=500)
        self._save(fig, "17_group_importance")

    def c18_entropy_regime_chart(self):
        """NEW v9: rolling entropy regime signal vs returns."""
        tp, _, tt, _ = self._get_test(); h_idx = 0
        p = tp[:, h_idx]; t2 = tt[:, h_idx]
        # Bin by prediction quartile
        q25, q75 = np.percentile(p, 25), np.percentile(p, 75)
        mask_hi = p >= q75; mask_lo = p <= q25
        x_all = list(range(len(p)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_all, y=t2, mode="lines",
                                 line=dict(color="white", width=1), name="Actual", opacity=0.4))
        fig.add_trace(go.Scatter(
            x=[i for i, m in enumerate(mask_hi) if m],
            y=[t2[i] for i, m in enumerate(mask_hi) if m],
            mode="markers", marker=dict(color="#59A14F", size=4), name="Top Quartile"))
        fig.add_trace(go.Scatter(
            x=[i for i, m in enumerate(mask_lo) if m],
            y=[t2[i] for i, m in enumerate(mask_lo) if m],
            mode="markers", marker=dict(color="#E15759", size=4), name="Bot Quartile"))
        fig.update_layout(title="Signal Quartile Overlay on Actual Returns (1d)",
                          template=DARK, height=450)
        self._save(fig, "18_entropy_regime_chart")

    def c19_per_horizon_attribution(self):
        _, _, _, m = self._get_test()
        attrs = ["IC", "RankIC", "DA", "Sharpe", "Calmar", "Q4Q1"]
        hors  = [str(h) for h in self.t.horizons]
        fig   = make_subplots(rows=2, cols=3,
                              subplot_titles=[f"{a}" for a in attrs])
        colors = px.colors.qualitative.Plotly
        for ai, attr in enumerate(attrs):
            vals = [m.get(f"{attr}_h{h}", 0.) for h in self.t.horizons]
            r, c = divmod(ai, 3)
            fig.add_trace(go.Bar(x=hors, y=vals,
                                 marker_color=[colors[i % len(colors)] for i in range(len(hors))],
                                 name=attr), row=r+1, col=c+1)
        fig.update_layout(title="Per-Horizon Attribution", template=DARK,
                          height=600, showlegend=False)
        self._save(fig, "19_per_horizon_attribution")

    def c20_summary_dashboard(self):
        _, _, _, m2 = self._get_test(); rows = []
        for key in ["IC_mean","RankIC_mean","DA_mean","Sharpe_mean"]:
            rows.append((key, f"{m2.get(key, 0.):.4f}"))
        for h2 in self.t.horizons:
            rows.append((f"{h2}d IC",     f"{m2.get(f'IC_h{h2}', 0.):.4f}"))
            rows.append((f"{h2}d DA",     f"{m2.get(f'DA_h{h2}', 0.):.3%}"))
            rows.append((f"{h2}d Sharpe", f"{m2.get(f'Sharpe_h{h2}', 0.):.3f}"))
        fig = make_subplots(rows=2, cols=2,
            specs=[[{"type":"table","colspan":2}, None], [{"type":"scatter"},{"type":"bar"}]],
            subplot_titles=["Test Metrics", "Loss Curves", "Val IC"],
            row_heights=[0.45, 0.55])
        fig.add_trace(go.Table(
            header=dict(values=["Metric","Value"], fill_color="#2d2d2d",
                        font=dict(color="white", size=13)),
            cells=dict(values=[[r[0] for r in rows], [r[1] for r in rows]],
                       fill_color=[["#1a1a2e"] * len(rows)],
                       font=dict(color="white", size=12))), row=1, col=1)
        h = self.h
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_loss"], name="Train",
                                 line=dict(color="#4E79A7", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_loss"],   name="Val",
                                 line=dict(color="#F28E2B", width=2, dash="dash")), row=2, col=1)
        fig.add_trace(go.Bar(x=h["epoch"], y=h["val_ic_mean"],
                             marker_color="#59A14F", name="IC Mean"), row=2, col=2)
        fig.update_layout(title="TFT v9 Multi-Horizon Performance Summary",
                          template=DARK, height=900)
        self._save(fig, "20_summary_dashboard")

    def generate_all(self):
        ts("Generating all 20 charts...")
        for fn in [self.c01_training_curves, self.c02_val_metrics, self.c03_vsn_importance,
                   self.c04_attn_heatmap, self.c05_pred_vs_actual, self.c06_ic_by_horizon,
                   self.c07_quantile_fan, self.c08_residual_dist, self.c09_feature_corr_matrix,
                   self.c10_ls_backtest, self.c11_rolling_ic, self.c12_quantile_calibration,
                   self.c13_decile_returns, self.c14_regime_ic, self.c15_horizon_sharpe,
                   self.c16_tta_uncertainty, self.c17_group_importance, self.c18_entropy_regime_chart,
                   self.c19_per_horizon_attribution, self.c20_summary_dashboard]:
            try: fn()
            except Exception as e: ts(f"WARN chart {fn.__name__}: {e}")
        ts(f"All 20 charts saved to {self.out}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ts("=== TFT v9 UNIFIED PIPELINE START ===")
    ts(f"Target: {TARGET_N_FEATURES} features | Horizons: {HORIZONS} | "
       f"Enc: {CFG.enc_len} | D: {CFG.hidden} | PurgeGap: {PURGE_GAP}")
    set_seed(CFG.seed)

    # ── Stage 1+2: feature engineering ───────────────────────────────────────
    run_pipeline()

    # ── Hardware ──────────────────────────────────────────────────────────────
    DEVICE, USE_AMP, N_GPUS = get_device()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    Path(CFG.out_dir).mkdir(parents=True, exist_ok=True)

    # ── DataLoader ────────────────────────────────────────────────────────────
    ts("Loading raw data...")
    loader = RawDataLoader(CFG.data_dir)
    if loader.n_unk == 0: ts("CRITICAL: no features found — check pipeline"); return

    ts("Building train dataset...")
    train_ds = MultiHorizonDataset(loader, split="train",
                                   mixup_alpha=CFG.mixup_alpha, cutmix_prob=CFG.cutmix_prob)
    gc.collect()
    if len(train_ds) == 0: ts("CRITICAL: empty train dataset"); return

    ts("Computing normalisation statistics (train only)...")
    norm_stats = train_ds.compute_norm_stats(); train_ds.apply_norm(norm_stats)
    ts("Saving norm_stats.npz ...")
    np.savez(os.path.join(CFG.out_dir, "norm_stats.npz"),
             unk_mu=norm_stats["unk_mu"], unk_sigma=norm_stats["unk_sigma"])
    ts("norm_stats saved.")

    ts("Building val dataset...")
    val_ds = MultiHorizonDataset(loader, split="val"); val_ds.apply_norm(norm_stats)
    ts(f"Val dataset built: {len(val_ds):,} windows"); gc.collect()

    ts("Building test dataset...")
    test_ds = MultiHorizonDataset(loader, split="test"); test_ds.apply_norm(norm_stats)
    ts(f"Test dataset built: {len(test_ds):,} windows"); gc.collect()

    ts(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # ── Determine optimal num_workers ─────────────────────────────────────────
    ts("Configuring DataLoaders ...")
    try:
        import multiprocessing
        nw = min(4, multiprocessing.cpu_count() // 2) if CFG.num_workers == 0 else CFG.num_workers
    except: nw = 0
    ts(f"  num_workers={nw}")
    # Kaggle TPU/GPU nodes: persistent_workers needs num_workers > 0
    pin = DEVICE.type == "cuda" and nw > 0
    kw = dict(collate_fn=fast_collate, num_workers=nw, pin_memory=pin,
              persistent_workers=(nw > 0))
    train_dl = DataLoader(train_ds, batch_size=CFG.batch, shuffle=True,  drop_last=True, **kw)
    val_dl   = DataLoader(val_ds,   batch_size=CFG.batch * 2, shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  batch_size=CFG.batch * 2, shuffle=False, **kw)
    ts(f"DataLoaders ready | train_batches={len(train_dl)} val_batches={len(val_dl)} "
       f"test_batches={len(test_dl)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    ts("Building TFT v9 model...")
    n_static = train_ds.n_static
    model = TFTv9(
        n_unk=loader.n_unk, n_kn=loader.n_kn, n_static=n_static,
        d_model=CFG.hidden, n_heads=CFG.heads, n_lstm_layers=CFG.n_lstm_layers,
        dropout=CFG.dropout, enc_len=CFG.enc_len, horizons=CFG.horizons,
        quantiles=CFG.quantiles, feat_group_ids=loader.feat_group_ids,
        n_groups=loader.n_groups, n_experts=CFG.n_experts, top_k_moe=CFG.top_k_moe)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ts(f"Model built: params={n_params:,} | unk={loader.n_unk} kn={loader.n_kn} "
       f"d={CFG.hidden} experts={CFG.n_experts} top_k={CFG.top_k_moe} horizons={CFG.horizons}")
    ts(f"RAM usage: {_check_ram_gb():.1f}GB / {RAM_LIMIT_GB}GB | "
       f"Disk free: {_free_space_gb('/tmp'):.1f}GB")

    # ── Trainer ───────────────────────────────────────────────────────────────
    ts("Initialising trainer ...")
    trainer = TFTTrainer(
        model, train_dl, val_dl, test_dl, norm_stats,
        loader.unk_names, loader.kn_names, CFG.quantiles, CFG.horizons,
        DEVICE, USE_AMP, N_GPUS)
    ts("Trainer ready — starting training loop ...")
    trainer.train()

    # ── Final evaluation ──────────────────────────────────────────────────────
    ts("Final test evaluation with TTA...")
    tl, tm, _, _, _ = trainer.eval_epoch(test_dl, tta=True)
    print(f"\n{'─'*65}"); ts(f"TEST LOSS: {tl:.5f}")
    for k, v in sorted(tm.items()):
        if isinstance(v, float): ts(f"  {k:40s}: {v:.5f}")
    print(f"{'─'*65}")

    # Save artefacts
    with open(os.path.join(CFG.out_dir, "test_metrics.json"), "w") as f:
        json.dump({k: (v if not (isinstance(v, float) and math.isnan(v)) else None)
                   for k, v in tm.items() if isinstance(v, float)}, f, indent=2)
    with open(os.path.join(CFG.out_dir, "feature_names.json"), "w") as f:
        json.dump({"unknown": loader.unk_names, "known": loader.kn_names}, f, indent=2)
    # Rolling IC CSV for downstream analysis
    try:
        ic_rows = []
        for k, v in tm.items():
            if k.startswith("IC_h") or k.startswith("RankIC_h") or k.startswith("DA_h"):
                ic_rows.append({"metric": k, "value": v})
        pd.DataFrame(ic_rows).to_csv(os.path.join(CFG.out_dir, "horizon_metrics.csv"), index=False)
    except: pass

    # Visualisations
    viz = TFTVisualizer(trainer, CFG.out_dir)
    viz.generate_all()

    total_elapsed = time.time() - _GLOBAL_START
    th = int(total_elapsed // 3600); tm2 = int((total_elapsed % 3600) // 60)
    ts2 = int(total_elapsed % 60)
    ts(f"=== TFT v9 COMPLETE — Total time: {th:02d}h{tm2:02d}m{ts2:02d}s ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as _top_e:
        print(f"\n{'='*65}", flush=True)
        print(f"FATAL ERROR in main(): {_top_e}", flush=True)
        print(traceback.format_exc(), flush=True)
        print(f"{'='*65}\n", flush=True)
        sys.exit(1)
