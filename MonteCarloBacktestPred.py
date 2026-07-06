

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import math
import time
import random
import argparse
import warnings
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy import optimize as scipy_optimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except Exception:
    _HAS_YFINANCE = False

warnings.filterwarnings("ignore")

EPS = 1e-9
TRADING_DAYS = 252
_GLOBAL_START = time.time()

DEFAULT_MODEL_PREDICTIONS_PATH = "./model_predictions.json"


# ─────────────────────────────────────────────────────────────────────────────
# Logging / misc helpers
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
class MCConfig:
    tickers: Tuple[str, ...] = ("AAPL",)
    start: str = "2015-01-01"
    end: Optional[str] = None                 # None = today
    out_dir: str = "./mc_results"

    engine: str = "ensemble"                  # gbm | jump | heston | garch | bootstrap | ensemble
    n_sims: int = 5000
    horizons: Tuple[int, ...] = (1, 5, 10, 21, 63, 126, 252)
    confidence_levels: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    var_alpha: float = 0.05

    calib_lookback_days: int = 756            # ~3y trailing window used to fit engines
    block_size: int = 20                      # bootstrap block length (trading days)
    student_t_df: float = 6.0                 # fat-tail innovations for GARCH engine
    risk_free_rate: float = 0.04

    # Rolling-origin backtest geometry
    backtest_n_origins: int = 40
    backtest_step: int = 10
    backtest_horizons: Tuple[int, ...] = (5, 21, 63)
    backtest_min_history: int = 504           # need >= 2y of history before first origin
    backtest_n_sims: int = 1000               # lighter sim count per backtest origin (speed)

    model_predictions_path: Optional[str] = None  # ticker -> {horizon: predicted_return}
    model_blend_weight: float = 0.5           # 0 = ignore model, 1 = fully trust model

    seed: int = 2024
    offline: bool = False                     # force synthetic data (skip yfinance)


# ─────────────────────────────────────────────────────────────────────────────
# Data retrieval — yfinance with graceful synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_price_series(ticker: str, start: str, end: Optional[str], seed: int) -> pd.DataFrame:
    """
    Generates a realistic-looking daily OHLCV series with regime rotation,
    volatility clustering (GARCH-style), and occasional jumps — used only
    when real data can't be retrieved, so the pipeline can still be
    exercised end to end.
    """
    rng = np.random.RandomState(abs(hash((ticker, seed))) % (2 ** 31))
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.today()
    dates = pd.bdate_range(start_dt, end_dt)
    n = len(dates)
    if n < 300:
        n = 300
        dates = pd.bdate_range(start_dt, periods=n, freq="B")

    # GARCH(1,1)-style conditional variance recursion for realistic vol clustering
    omega, alpha, beta = 1e-6, 0.08, 0.90
    h = np.zeros(n)
    h[0] = omega / max(1 - alpha - beta, 1e-3)
    eps = rng.standard_t(6, size=n)
    ret = np.zeros(n)
    mu_daily = rng.uniform(0.0002, 0.0005)
    for i in range(1, n):
        # Clip the previous day's squared return feeding the GARCH recursion —
        # otherwise a single heavy-tailed Student-t draw can create a runaway
        # positive-feedback loop (huge return -> huge variance -> huger return).
        prev_ret_sq = min(ret[i - 1] ** 2, 0.04)
        h[i] = omega + alpha * prev_ret_sq + beta * h[i - 1]
        h[i] = min(h[i], 0.01)  # cap daily variance at (10% daily vol)^2
        ret[i] = mu_daily + math.sqrt(max(h[i], 1e-10)) * eps[i]
        if rng.rand() < 0.01:  # occasional jump
            ret[i] += rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)
        ret[i] = float(np.clip(ret[i], -0.20, 0.20))  # sane daily-move guard rail

    close = 100.0 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.lognormal(mean=14.0, sigma=0.6, size=n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class MarketDataLoader:
    """Thin wrapper around yfinance with retries, on-disk caching, and a
    synthetic fallback so the module never hard-fails for lack of network."""

    def __init__(self, cache_dir: str = "/tmp/mc_backtest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        safe = f"{ticker}_{start}_{end}".replace("/", "-")
        return self.cache_dir / f"{safe}.parquet"

    def fetch(self, ticker: str, start: str, end: Optional[str], offline: bool,
             seed: int, retries: int = 3) -> Tuple[pd.DataFrame, bool]:
        """Returns (dataframe, is_synthetic)."""
        end_str = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        cache_p = self._cache_path(ticker, start, end_str)

        if offline or not _HAS_YFINANCE:
            reason = "offline flag set" if offline else "yfinance not installed"
            ts(f"[{ticker}] {reason} — using synthetic fallback data")
            return _synthetic_price_series(ticker, start, end_str, seed), True

        if cache_p.exists():
            try:
                df = pd.read_parquet(cache_p)
                if len(df) > 50:
                    ts(f"[{ticker}] loaded {len(df)} rows from cache")
                    return df, False
            except Exception:
                pass

        for attempt in range(1, retries + 1):
            try:
                df = yf.download(ticker, start=start, end=end_str, progress=False,
                                 auto_adjust=True, threads=False)
                if df is None or df.empty:
                    raise RuntimeError("empty dataframe returned")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                try:
                    df.to_parquet(cache_p)
                except Exception:
                    pass
                ts(f"[{ticker}] downloaded {len(df)} rows from yfinance")
                return df, False
            except Exception as e:
                ts(f"[{ticker}] yfinance attempt {attempt}/{retries} failed: {e}")
                time.sleep(1.5 * attempt)

        ts(f"[{ticker}] all yfinance attempts failed — using synthetic fallback data")
        return _synthetic_price_series(ticker, start, end_str, seed), True


def log_returns_from_close(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    r = np.zeros(len(close))
    r[1:] = np.log(np.clip(close[1:], EPS, None) / np.clip(close[:-1], EPS, None))
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Calibration: estimate parameters for each stochastic engine from history
# ─────────────────────────────────────────────────────────────────────────────
def estimate_gbm_params(log_ret: np.ndarray) -> Dict[str, float]:
    mu_daily = float(np.mean(log_ret))
    sigma_daily = float(np.std(log_ret, ddof=1))
    return {"mu": mu_daily * TRADING_DAYS, "sigma": sigma_daily * math.sqrt(TRADING_DAYS)}


def estimate_merton_jump_params(log_ret: np.ndarray, jump_z_thresh: float = 3.0) -> Dict[str, float]:
    """
    Simple (method-of-moments) jump/diffusion split: returns beyond
    `jump_z_thresh` standard deviations are classified as jumps; the
    remainder calibrates the diffusive GBM component.
    """
    mu_all = np.mean(log_ret)
    sd_all = np.std(log_ret, ddof=1) + EPS
    z = (log_ret - mu_all) / sd_all
    jump_mask = np.abs(z) > jump_z_thresh
    n = len(log_ret)
    n_jumps = int(jump_mask.sum())
    lam_daily = n_jumps / max(n, 1)

    diffusive = log_ret[~jump_mask]
    if len(diffusive) < 30:
        diffusive = log_ret
    mu_d = float(np.mean(diffusive))
    sigma_d = float(np.std(diffusive, ddof=1))

    if n_jumps >= 3:
        jump_vals = log_ret[jump_mask]
        jump_mean = float(np.mean(jump_vals))
        jump_std = float(np.std(jump_vals, ddof=1)) if n_jumps > 1 else float(sigma_d * 2)
    else:
        jump_mean, jump_std = 0.0, sigma_d * 2

    return {
        "mu": mu_d * TRADING_DAYS,
        "sigma": sigma_d * math.sqrt(TRADING_DAYS),
        "lam": lam_daily * TRADING_DAYS,       # annualised jump intensity
        "jump_mean": jump_mean,
        "jump_std": max(jump_std, 1e-4),
    }


def _garch11_neg_loglik(params, ret):
    omega, alpha, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999 or nu <= 2.05:
        return 1e10
    n = len(ret)
    h = np.empty(n)
    h[0] = np.var(ret)
    ll = 0.0
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log((nu - 2) * math.pi)
    for i in range(1, n):
        h[i] = omega + alpha * ret[i - 1] ** 2 + beta * h[i - 1]
        h[i] = max(h[i], 1e-12)
        z2 = ret[i] ** 2 / h[i]
        ll += c - 0.5 * math.log(h[i]) - ((nu + 1) / 2) * math.log(1 + z2 / (nu - 2))
    return -ll


def fit_garch11(log_ret: np.ndarray) -> Dict[str, float]:
    """
    Lightweight GARCH(1,1) with Student-t innovations, fit via numerical
    MLE (no external `arch` dependency required). Falls back to a
    variance-targeted guess if the optimizer fails to converge cleanly.
    """
    ret = log_ret - np.mean(log_ret)
    var_target = np.var(ret)
    x0 = np.array([var_target * 0.05, 0.08, 0.85, 8.0])
    bounds = [(1e-10, var_target), (1e-4, 0.3), (0.5, 0.98), (2.5, 30.0)]
    try:
        res = scipy_optimize.minimize(
            _garch11_neg_loglik, x0, args=(ret,), method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 200})
        omega, alpha, beta, nu = res.x
        if not (0 < alpha + beta < 1):
            raise ValueError("non-stationary fit")
    except Exception:
        omega = var_target * 0.05
        alpha, beta, nu = 0.08, 0.85, 8.0
    mu_daily = float(np.mean(log_ret))
    return {"mu": mu_daily, "omega": float(omega), "alpha": float(alpha),
           "beta": float(beta), "nu": float(nu)}


def estimate_heston_params(log_ret: np.ndarray) -> Dict[str, float]:
    """
    Approximate Heston calibration via realised-variance proxies (a full
    options-implied calibration is out of scope here — this uses a
    method-of-moments style fit against the historical variance process,
    which is standard practice for a physical-measure Monte Carlo engine).
    """
    mu_daily = float(np.mean(log_ret))
    sq = log_ret ** 2
    v0 = float(np.var(log_ret[-20:])) if len(log_ret) >= 20 else float(np.var(log_ret))
    theta = float(np.var(log_ret))                      # long-run variance
    # AR(1) on squared returns as a variance-mean-reversion proxy
    if len(sq) > 30:
        x = sq[:-1] - sq[:-1].mean()
        y = sq[1:] - sq[1:].mean()
        denom = float(np.dot(x, x)) + EPS
        phi = float(np.clip(np.dot(x, y) / denom, -0.98, 0.98))
    else:
        phi = 0.9
    kappa = max(-math.log(max(phi, 1e-4)) * TRADING_DAYS, 1.0)
    resid_std = float(np.std(sq[1:] - phi * sq[:-1])) if len(sq) > 30 else theta * 0.5
    xi = float(np.clip(resid_std * math.sqrt(TRADING_DAYS), 0.05, 3.0))  # vol-of-vol
    rho = float(np.clip(np.corrcoef(log_ret[:-1], sq[1:] - sq[:-1])[0, 1], -0.95, 0.0)) \
        if len(log_ret) > 30 else -0.4
    return {"mu": mu_daily * TRADING_DAYS, "kappa": kappa, "theta": theta * TRADING_DAYS,
           "xi": xi, "rho": rho, "v0": v0 * TRADING_DAYS}


def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """Classic R/S-statistic Hurst exponent estimate."""
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    max_lag = min(max_lag, n // 2)
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diffs = series[lag:] - series[:-lag]
        tau.append(np.std(diffs) + EPS)
    if len(tau) < 2:
        return 0.5
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Historical block bootstrap sampler (circular block bootstrap)
# ─────────────────────────────────────────────────────────────────────────────
class BlockBootstrapSampler:
    """
    Resamples overlapping blocks of historical log-returns to build
    synthetic forward paths that preserve empirical autocorrelation,
    volatility clustering, and fat tails by construction (no distributional
    assumption is imposed — the historical data speaks for itself).
    """
    def __init__(self, hist_log_ret: np.ndarray, block_size: int, rng: np.random.RandomState):
        self.hist = hist_log_ret
        self.block_size = max(block_size, 1)
        self.rng = rng
        self.n = len(hist_log_ret)

    def sample_path(self, n_days: int) -> np.ndarray:
        out = np.empty(n_days)
        filled = 0
        while filled < n_days:
            start = self.rng.randint(0, self.n)
            block = np.take(self.hist, range(start, start + self.block_size), mode="wrap")
            take = min(self.block_size, n_days - filled)
            out[filled:filled + take] = block[:take]
            filled += take
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo simulation engines — every engine returns a [n_paths, n_days]
# matrix of simulated DAILY LOG RETURNS (never prices directly), so engines
# compose cleanly and price paths are reconstructed once, centrally.
# ─────────────────────────────────────────────────────────────────────────────
class MCEngine:
    name = "base"

    def simulate_log_returns(self, n_days: int, n_paths: int, rng: np.random.RandomState) -> np.ndarray:
        raise NotImplementedError


class GBMEngine(MCEngine):
    name = "gbm"

    def __init__(self, mu: float, sigma: float):
        self.mu_daily = mu / TRADING_DAYS
        self.sigma_daily = sigma / math.sqrt(TRADING_DAYS)

    def simulate_log_returns(self, n_days, n_paths, rng):
        drift = self.mu_daily - 0.5 * self.sigma_daily ** 2
        z = rng.standard_normal((n_paths, n_days))
        return drift + self.sigma_daily * z


class MertonJumpEngine(MCEngine):
    name = "jump"

    def __init__(self, mu, sigma, lam, jump_mean, jump_std):
        self.mu_daily = mu / TRADING_DAYS
        self.sigma_daily = sigma / math.sqrt(TRADING_DAYS)
        self.lam_daily = lam / TRADING_DAYS
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def simulate_log_returns(self, n_days, n_paths, rng):
        drift = self.mu_daily - 0.5 * self.sigma_daily ** 2
        z = rng.standard_normal((n_paths, n_days))
        diffusion = drift + self.sigma_daily * z
        n_jumps = rng.poisson(self.lam_daily, size=(n_paths, n_days))
        jump_sizes = rng.normal(self.jump_mean, self.jump_std, size=(n_paths, n_days))
        return diffusion + n_jumps * jump_sizes


class HestonEngine(MCEngine):
    """Euler discretisation with full-truncation scheme for variance positivity."""
    name = "heston"

    def __init__(self, mu, kappa, theta, xi, rho, v0):
        self.mu_daily = mu / TRADING_DAYS
        self.kappa_daily = kappa / TRADING_DAYS
        self.theta_daily = theta / TRADING_DAYS
        self.xi_daily = xi / math.sqrt(TRADING_DAYS)
        self.rho = rho
        self.v0_daily = v0 / TRADING_DAYS

    def simulate_log_returns(self, n_days, n_paths, rng):
        v = np.full(n_paths, max(self.v0_daily, 1e-8))
        out = np.empty((n_paths, n_days))
        v_cap = max(self.theta_daily * 8, 0.01)
        for t in range(n_days):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            zv = z1
            zs = self.rho * z1 + math.sqrt(max(1 - self.rho ** 2, 0.0)) * z2
            v_pos = np.clip(v, 0, v_cap)
            v_next = (v + self.kappa_daily * (self.theta_daily - v_pos)
                     + self.xi_daily * np.sqrt(v_pos) * zv)
            v_next = np.clip(v_next, 0, v_cap)
            ret = self.mu_daily - 0.5 * v_pos + np.sqrt(v_pos) * zs
            out[:, t] = np.clip(ret, -0.20, 0.20)
            v = v_next
        return out


class GARCHEngine(MCEngine):
    name = "garch"

    def __init__(self, mu, omega, alpha, beta, nu=6.0, h0=None):
        self.mu_daily = mu
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.nu = max(nu, 2.5)
        self.h0 = h0 if h0 is not None else omega / max(1 - alpha - beta, 1e-3)

    def simulate_log_returns(self, n_days, n_paths, rng):
        h = np.full(n_paths, max(self.h0, 1e-10))
        out = np.empty((n_paths, n_days))
        t_scale = math.sqrt((self.nu - 2) / self.nu) if self.nu > 2 else 1.0
        for t in range(n_days):
            z = rng.standard_t(self.nu, size=n_paths) * t_scale
            eps = np.sqrt(np.clip(h, 1e-12, 0.01)) * z
            eps = np.clip(eps, -0.20, 0.20)
            out[:, t] = self.mu_daily + eps
            h = self.omega + self.alpha * np.clip(eps ** 2, 0, 0.04) + self.beta * h
            h = np.clip(h, 1e-12, 0.01)
        return out


class BootstrapEngine(MCEngine):
    name = "bootstrap"

    def __init__(self, sampler: BlockBootstrapSampler):
        self.sampler = sampler

    def simulate_log_returns(self, n_days, n_paths, rng):
        out = np.empty((n_paths, n_days))
        for p in range(n_paths):
            out[p] = self.sampler.sample_path(n_days)
        return out


class EnsembleEngine(MCEngine):
    """Mixture ensemble: each simulated path is generated by one engine,
    drawn according to `weights` — this yields a forecast distribution
    that blends parametric and non-parametric (bootstrap) views."""
    name = "ensemble"

    def __init__(self, engines: List[MCEngine], weights: Optional[List[float]] = None):
        self.engines = engines
        w = np.array(weights if weights is not None else [1.0] * len(engines), dtype=np.float64)
        self.weights = w / w.sum()

    def simulate_log_returns(self, n_days, n_paths, rng):
        counts = rng.multinomial(n_paths, self.weights)
        chunks = []
        for engine, c in zip(self.engines, counts):
            if c > 0:
                chunks.append(engine.simulate_log_returns(n_days, c, rng))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((n_paths, n_days))


def build_engine(name: str, log_ret: np.ndarray, cfg: MCConfig,
                 rng: np.random.RandomState) -> MCEngine:
    name = name.lower()
    if name == "gbm":
        p = estimate_gbm_params(log_ret)
        return GBMEngine(p["mu"], p["sigma"])
    if name == "jump":
        p = estimate_merton_jump_params(log_ret)
        return MertonJumpEngine(p["mu"], p["sigma"], p["lam"], p["jump_mean"], p["jump_std"])
    if name == "heston":
        p = estimate_heston_params(log_ret)
        return HestonEngine(p["mu"], p["kappa"], p["theta"], p["xi"], p["rho"], p["v0"])
    if name == "garch":
        p = fit_garch11(log_ret)
        return GARCHEngine(p["mu"], p["omega"], p["alpha"], p["beta"], p["nu"])
    if name == "bootstrap":
        sampler = BlockBootstrapSampler(log_ret, cfg.block_size, rng)
        return BootstrapEngine(sampler)
    if name == "ensemble":
        sub = [
            build_engine("gbm", log_ret, cfg, rng),
            build_engine("jump", log_ret, cfg, rng),
            build_engine("heston", log_ret, cfg, rng),
            build_engine("garch", log_ret, cfg, rng),
            build_engine("bootstrap", log_ret, cfg, rng),
        ]
        # Bootstrap and GARCH get more weight — they best reproduce fat
        # tails / clustering without distributional assumptions.
        return EnsembleEngine(sub, weights=[0.15, 0.15, 0.15, 0.25, 0.30])
    raise ValueError(f"unknown engine: {name}")


def simulate_price_paths(engine: MCEngine, S0: float, n_days: int, n_paths: int,
                         rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (price_paths [n_paths, n_days+1], log_return_paths [n_paths, n_days])."""
    log_rets = engine.simulate_log_returns(n_days, n_paths, rng)
    log_rets = np.clip(log_rets, -0.5, 0.5)  # guard against pathological single-day blowups
    cum = np.cumsum(log_rets, axis=1)
    prices = np.empty((n_paths, n_days + 1))
    prices[:, 0] = S0
    prices[:, 1:] = S0 * np.exp(cum)
    return prices, log_rets


# ─────────────────────────────────────────────────────────────────────────────
# Realism validation ("Reality Check")
# ─────────────────────────────────────────────────────────────────────────────
def ljung_box_stat(x: np.ndarray, lags: int = 10) -> Tuple[float, float]:
    """Ljung-Box Q statistic + p-value for autocorrelation up to `lags`."""
    n = len(x)
    x = x - np.mean(x)
    acf = []
    denom = np.dot(x, x)
    for k in range(1, lags + 1):
        num = np.dot(x[:-k], x[k:])
        acf.append(num / (denom + EPS))
    acf = np.array(acf)
    q = n * (n + 2) * np.sum((acf ** 2) / (n - np.arange(1, lags + 1)))
    p_value = 1 - scipy_stats.chi2.cdf(q, df=lags)
    return float(q), float(p_value)


@dataclass
class RealismReport:
    ticker: str
    hist_mean_ann: float
    sim_mean_ann: float
    hist_vol_ann: float
    sim_vol_ann: float
    hist_skew: float
    sim_skew: float
    hist_kurt: float
    sim_kurt: float
    hist_hurst: float
    sim_hurst: float
    hist_ljungbox_p: float          # returns ACF — should be near-white-noise (high p)
    sim_ljungbox_p: float
    hist_sq_ljungbox_p: float        # squared returns ACF — should show clustering (low p)
    sim_sq_ljungbox_p: float
    checks: Dict[str, bool] = field(default_factory=dict)
    realism_score: float = 0.0

    def to_dict(self):
        return asdict(self)


class RealismValidator:
    """
    Compares one simulated path's return series against the true historical
    series across the stylized facts that define "realistic" equity returns:
    volatility level, fat tails, weak linear autocorrelation, strong
    volatility clustering (ARCH effects), and long-memory (Hurst) behaviour.
    """
    def __init__(self, tolerance: Dict[str, float] = None):
        self.tol = tolerance or {
            "vol_ratio": 0.45,      # sim annualised vol within +/-45% of historical
            "kurt_ratio": 0.6,      # sim excess kurtosis within 60% (fat tails preserved)
            "hurst_abs_diff": 0.18,
        }

    def validate(self, ticker: str, hist_log_ret: np.ndarray, sim_log_ret: np.ndarray) -> RealismReport:
        hist_mean_ann = float(np.mean(hist_log_ret) * TRADING_DAYS)
        sim_mean_ann = float(np.mean(sim_log_ret) * TRADING_DAYS)
        hist_vol_ann = float(np.std(hist_log_ret, ddof=1) * math.sqrt(TRADING_DAYS))
        sim_vol_ann = float(np.std(sim_log_ret, ddof=1) * math.sqrt(TRADING_DAYS))
        hist_skew = float(scipy_stats.skew(hist_log_ret))
        sim_skew = float(scipy_stats.skew(sim_log_ret))
        hist_kurt = float(scipy_stats.kurtosis(hist_log_ret))     # excess kurtosis
        sim_kurt = float(scipy_stats.kurtosis(sim_log_ret))
        hist_hurst = hurst_exponent(np.cumsum(hist_log_ret))
        sim_hurst = hurst_exponent(np.cumsum(sim_log_ret))
        _, hist_lb_p = ljung_box_stat(hist_log_ret, lags=10)
        _, sim_lb_p = ljung_box_stat(sim_log_ret, lags=10)
        _, hist_sq_lb_p = ljung_box_stat(hist_log_ret ** 2, lags=10)
        _, sim_sq_lb_p = ljung_box_stat(sim_log_ret ** 2, lags=10)

        checks = {}
        vol_ratio = abs(sim_vol_ann - hist_vol_ann) / (hist_vol_ann + EPS)
        checks["volatility_within_tolerance"] = vol_ratio <= self.tol["vol_ratio"]

        kurt_ratio = abs(sim_kurt - hist_kurt) / (abs(hist_kurt) + 1.0)
        checks["fat_tails_preserved"] = kurt_ratio <= self.tol["kurt_ratio"] or sim_kurt >= hist_kurt * 0.4

        checks["hurst_within_tolerance"] = abs(sim_hurst - hist_hurst) <= self.tol["hurst_abs_diff"]
        checks["returns_near_white_noise"] = sim_lb_p > 0.01          # weak linear autocorr, as expected
        checks["vol_clustering_present"] = sim_sq_lb_p < 0.20          # ARCH effects should show up

        score = 100.0 * sum(checks.values()) / len(checks)

        return RealismReport(
            ticker=ticker, hist_mean_ann=hist_mean_ann, sim_mean_ann=sim_mean_ann,
            hist_vol_ann=hist_vol_ann, sim_vol_ann=sim_vol_ann,
            hist_skew=hist_skew, sim_skew=sim_skew, hist_kurt=hist_kurt, sim_kurt=sim_kurt,
            hist_hurst=hist_hurst, sim_hurst=sim_hurst,
            hist_ljungbox_p=hist_lb_p, sim_ljungbox_p=sim_lb_p,
            hist_sq_ljungbox_p=hist_sq_lb_p, sim_sq_ljungbox_p=sim_sq_lb_p,
            checks=checks, realism_score=score)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-horizon prediction summary (+ optional model blending)
# ─────────────────────────────────────────────────────────────────────────────
def _var_cvar(returns_at_h: np.ndarray, alpha: float) -> Tuple[float, float]:
    var = float(np.percentile(returns_at_h, alpha * 100))
    tail = returns_at_h[returns_at_h <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


def summarize_horizon(price_paths: np.ndarray, S0: float, h: int, cfg: MCConfig,
                      model_pred: Optional[float] = None) -> Dict[str, float]:
    terminal = price_paths[:, h]
    rets = terminal / S0 - 1.0

    if model_pred is not None and cfg.model_blend_weight > 0:
        # Precision-agnostic linear blend: shift the whole simulated
        # distribution so its mean moves toward the model's point forecast,
        # weighted by `model_blend_weight`, while preserving the MC-implied
        # SHAPE (spread, skew, tails) of the distribution.
        w = float(np.clip(cfg.model_blend_weight, 0.0, 1.0))
        mc_mean = float(np.mean(rets))
        target_mean = (1 - w) * mc_mean + w * model_pred
        rets = rets - mc_mean + target_mean

    mean_r = float(np.mean(rets))
    median_r = float(np.median(rets))
    std_r = float(np.std(rets, ddof=1))
    skew_r = float(scipy_stats.skew(rets))
    kurt_r = float(scipy_stats.kurtosis(rets))
    prob_pos = float(np.mean(rets > 0))
    var_, cvar_ = _var_cvar(rets, cfg.var_alpha)
    ann_factor = math.sqrt(TRADING_DAYS / max(h, 1))
    sharpe = float((mean_r - cfg.risk_free_rate * h / TRADING_DAYS) / (std_r + EPS) * ann_factor)

    out = {
        "horizon_days": h, "mean_return": mean_r, "median_return": median_r,
        "std_return": std_r, "skew": skew_r, "excess_kurtosis": kurt_r,
        "prob_positive": prob_pos, f"VaR_{int(cfg.var_alpha*100)}": var_,
        f"CVaR_{int(cfg.var_alpha*100)}": cvar_, "annualized_sharpe": sharpe,
        "model_blended": model_pred is not None,
    }
    for cl in cfg.confidence_levels:
        out[f"pctl_{int(cl*100)}"] = float(np.percentile(rets, cl * 100))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo rolling-origin backtester (calibration quality check)
# ─────────────────────────────────────────────────────────────────────────────
def crps_empirical(samples: np.ndarray, y: float) -> float:
    """Unbiased empirical CRPS (energy-score form): E|X-y| - 0.5*E|X-X'|."""
    term1 = np.mean(np.abs(samples - y))
    m = min(len(samples), 500)  # subsample the O(n^2) term for speed
    sub = np.random.choice(samples, size=m, replace=False) if len(samples) > m else samples
    term2 = 0.5 * np.mean(np.abs(sub[:, None] - sub[None, :]))
    return float(term1 - term2)


def kupiec_pof_test(breaches: int, n_obs: int, alpha: float) -> Tuple[float, float]:
    """Kupiec proportion-of-failures likelihood-ratio test for VaR backtesting."""
    if n_obs == 0:
        return 0.0, 1.0
    pi_hat = breaches / n_obs
    pi_hat = min(max(pi_hat, 1e-6), 1 - 1e-6)
    ll_null = (n_obs - breaches) * math.log(1 - alpha) + breaches * math.log(alpha)
    ll_alt = (n_obs - breaches) * math.log(1 - pi_hat) + breaches * math.log(pi_hat)
    lr = -2 * (ll_null - ll_alt)
    p_value = 1 - scipy_stats.chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


class MonteCarloBacktester:
    """
    At each of several historical "origin" dates, calibrates the chosen MC
    engine using ONLY data available as of that date (strictly causal),
    simulates forward to each backtest horizon, and checks whether the
    REALIZED future return (which we can see, since this is history) fell
    where the simulated distribution said it should.

    Aggregated over many origins this yields:
      - PIT (probability integral transform) values per horizon: should be
        ~Uniform(0,1) if the model's predictive distribution is well
        calibrated (checked with a Kolmogorov-Smirnov test).
      - Empirical coverage of each nominal confidence interval vs its
        advertised level.
      - Kupiec POF test for the VaR breach rate.
      - Mean CRPS (lower is better) and forecast bias.
    """
    def __init__(self, cfg: MCConfig):
        self.cfg = cfg

    def run(self, close: np.ndarray, engine_name: str, rng: np.random.RandomState) -> Dict:
        cfg = self.cfg
        n = len(close)
        max_h = max(cfg.backtest_horizons)
        first_origin = cfg.backtest_min_history
        last_origin = n - max_h - 1
        if last_origin <= first_origin:
            ts("  [Backtest] insufficient history for requested horizons — skipping")
            return {}

        origins = list(range(first_origin, last_origin, cfg.backtest_step))
        if cfg.backtest_n_origins:
            origins = origins[-cfg.backtest_n_origins:]

        per_h_pit = {h: [] for h in cfg.backtest_horizons}
        per_h_breach = {h: 0 for h in cfg.backtest_horizons}
        per_h_crps = {h: [] for h in cfg.backtest_horizons}
        per_h_bias = {h: [] for h in cfg.backtest_horizons}
        per_h_coverage = {h: {cl: [] for cl in cfg.confidence_levels} for h in cfg.backtest_horizons}

        run_start = time.time()
        for oi, o in enumerate(origins):
            calib_start = max(0, o - cfg.calib_lookback_days)
            hist_close = close[calib_start:o + 1]
            log_ret = log_returns_from_close(hist_close)[1:]
            if len(log_ret) < 60:
                continue
            try:
                engine = build_engine(engine_name, log_ret, cfg, rng)
                S0 = float(close[o])
                paths, _ = simulate_price_paths(engine, S0, max_h, cfg.backtest_n_sims, rng)
            except Exception as e:
                ts(f"  [Backtest] origin {o} calibration failed: {e}")
                continue

            for h in cfg.backtest_horizons:
                realized_price = close[o + h]
                realized_ret = realized_price / S0 - 1.0
                sim_rets = paths[:, h] / S0 - 1.0

                pit = float(np.mean(sim_rets <= realized_ret))
                per_h_pit[h].append(pit)

                var_, _ = _var_cvar(sim_rets, cfg.var_alpha)
                if realized_ret < var_:
                    per_h_breach[h] += 1

                per_h_crps[h].append(crps_empirical(sim_rets, realized_ret))
                per_h_bias[h].append(float(np.mean(sim_rets)) - realized_ret)

                for cl in cfg.confidence_levels:
                    lo = np.percentile(sim_rets, (1 - cl) / 2 * 100)
                    hi = np.percentile(sim_rets, (1 - (1 - cl) / 2) * 100)
                    per_h_coverage[h][cl].append(1.0 if lo <= realized_ret <= hi else 0.0)

            if (oi + 1) % max(1, len(origins) // 5) == 0:
                ts(f"  [Backtest] origin {oi+1}/{len(origins)} | "
                   f"ETA={eta((oi+1)/len(origins), run_start)}")

        report = {"n_origins": len(origins), "per_horizon": {}}
        for h in cfg.backtest_horizons:
            pit_vals = np.array(per_h_pit[h])
            if len(pit_vals) < 5:
                continue
            ks_stat, ks_p = scipy_stats.kstest(pit_vals, "uniform")
            n_obs = len(pit_vals)
            breaches = per_h_breach[h]
            lr, kupiec_p = kupiec_pof_test(breaches, n_obs, cfg.var_alpha)
            coverage = {
                f"coverage_{int(cl*100)}": {
                    "nominal": cl, "empirical": float(np.mean(per_h_coverage[h][cl]))
                } for cl in cfg.confidence_levels
            }
            report["per_horizon"][str(h)] = {
                "n_obs": n_obs,
                "pit_ks_stat": float(ks_stat), "pit_ks_pvalue": float(ks_p),
                "pit_calibrated": bool(ks_p > 0.05),
                "var_breach_rate": breaches / n_obs, "var_nominal_rate": cfg.var_alpha,
                "kupiec_lr_stat": lr, "kupiec_pvalue": kupiec_p,
                "var_calibrated": bool(kupiec_p > 0.05),
                "mean_crps": float(np.mean(per_h_crps[h])),
                "mean_bias": float(np.mean(per_h_bias[h])),
                "coverage": coverage,
                "pit_values": pit_vals.tolist(),
            }
        return report


# ─────────────────────────────────────────────────────────────────────────────
# Model-prediction blending helper (pairs with walk_forward_validation.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_model_predictions(path: Optional[str]) -> Dict[str, Dict[int, float]]:
    """
    Expected JSON schema:
        {"AAPL": {"1": 0.001, "5": 0.004, "21": 0.012}, "MSFT": {...}}
    Missing file / ticker / horizon simply means "no model view available"
    and the module falls back to the pure Monte Carlo forecast for that
    slice — this is the "if the model isn't run" path the user described.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        ts(f"WARN: model predictions file not found at {path} — using pure Monte Carlo forecasts")
        return {}
    try:
        with open(p) as fh:
            raw = json.load(fh)
        return {tkr: {int(h): v for h, v in hz.items()} for tkr, hz in raw.items()}
    except Exception as e:
        ts(f"WARN: failed to parse model predictions ({e}) — using pure Monte Carlo forecasts")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_fan_chart(hist_close: np.ndarray, sim_paths: np.ndarray, ticker: str, out_path: Path):
    n_hist = len(hist_close)
    n_fwd = sim_paths.shape[1] - 1
    x_hist = np.arange(-n_hist, 0)
    x_fwd = np.arange(0, n_fwd + 1)

    pctl_levels = [5, 25, 50, 75, 95]
    bands = {p: np.percentile(sim_paths, p, axis=0) for p in pctl_levels}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x_hist, hist_close, color="#2c2c2c", linewidth=1.2, label="Historical close")
    ax.fill_between(x_fwd, bands[5], bands[95], color="#4E79A7", alpha=0.15, label="5-95 pct band")
    ax.fill_between(x_fwd, bands[25], bands[75], color="#4E79A7", alpha=0.30, label="25-75 pct band")
    ax.plot(x_fwd, bands[50], color="#F28E2B", linewidth=1.6, label="Median simulated path")
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_title(f"{ticker} — Monte Carlo Fan Chart")
    ax.set_xlabel("Trading days relative to simulation origin")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_terminal_distributions(price_paths: np.ndarray, S0: float, horizons: Tuple[int, ...],
                                ticker: str, out_path: Path):
    n = len(horizons)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for i, h in enumerate(horizons):
        rets = price_paths[:, h] / S0 - 1.0
        axes[i].hist(rets, bins=60, color="#4E79A7", alpha=0.85)
        axes[i].axvline(0, color="grey", linestyle=":", linewidth=1)
        axes[i].axvline(np.median(rets), color="#F28E2B", linewidth=1.5)
        axes[i].set_title(f"{h}d return distribution")
    for j in range(len(horizons), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{ticker} — Simulated Terminal Return Distributions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pit_calibration(report: Dict, ticker: str, out_path: Path):
    horizons = list(report.get("per_horizon", {}).keys())
    if not horizons:
        return
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.6))
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, horizons):
        pit = np.array(report["per_horizon"][h]["pit_values"])
        ax.hist(pit, bins=15, range=(0, 1), color="#59A14F", alpha=0.85, density=True)
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.5)
        p = report["per_horizon"][h]["pit_ks_pvalue"]
        ax.set_title(f"h={h}d PIT (KS p={p:.3f})")
        ax.set_xlabel("PIT value")
    fig.suptitle(f"{ticker} — Backtest PIT Calibration (flat = well-calibrated)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_coverage_bars(report: Dict, ticker: str, out_path: Path):
    horizons = list(report.get("per_horizon", {}).keys())
    if not horizons:
        return
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.6))
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, horizons):
        cov = report["per_horizon"][h]["coverage"]
        labels = sorted(cov.keys(), key=lambda k: cov[k]["nominal"])
        nominal = [cov[k]["nominal"] for k in labels]
        empirical = [cov[k]["empirical"] for k in labels]
        x = np.arange(len(labels))
        ax.bar(x - 0.18, nominal, width=0.35, label="Nominal", color="#4E79A7")
        ax.bar(x + 0.18, empirical, width=0.35, label="Empirical", color="#F28E2B")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(n*100)}%" for n in nominal])
        ax.set_title(f"h={h}d coverage")
        ax.legend(fontsize=7)
    fig.suptitle(f"{ticker} — Nominal vs Empirical Interval Coverage", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_realism_diagnostics(hist_log_ret: np.ndarray, sim_log_ret: np.ndarray,
                             ticker: str, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].hist(hist_log_ret, bins=80, alpha=0.6, density=True, label="Historical", color="#4E79A7")
    axes[0, 0].hist(sim_log_ret, bins=80, alpha=0.6, density=True, label="Simulated", color="#F28E2B")
    axes[0, 0].set_title("Daily log-return distribution")
    axes[0, 0].legend(fontsize=8)

    def _acf(x, lags=20):
        x = x - x.mean()
        denom = np.dot(x, x) + EPS
        return [np.dot(x[:-k], x[k:]) / denom for k in range(1, lags + 1)]

    lags = range(1, 21)
    axes[0, 1].bar(np.array(lags) - 0.15, _acf(hist_log_ret), width=0.3, label="Historical", color="#4E79A7")
    axes[0, 1].bar(np.array(lags) + 0.15, _acf(sim_log_ret), width=0.3, label="Simulated", color="#F28E2B")
    axes[0, 1].set_title("Return ACF (should be ~0 both)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(np.array(lags) - 0.15, _acf(hist_log_ret ** 2), width=0.3, label="Historical", color="#4E79A7")
    axes[1, 0].bar(np.array(lags) + 0.15, _acf(sim_log_ret ** 2), width=0.3, label="Simulated", color="#F28E2B")
    axes[1, 0].set_title("Squared-return ACF (vol clustering)")
    axes[1, 0].legend(fontsize=8)

    qs = np.linspace(0.01, 0.99, 50)
    axes[1, 1].plot(np.percentile(hist_log_ret, qs * 100), np.percentile(sim_log_ret, qs * 100),
                    marker="o", markersize=2, linestyle="none", color="#59A14F")
    lims = [min(hist_log_ret.min(), sim_log_ret.min()), max(hist_log_ret.max(), sim_log_ret.max())]
    axes[1, 1].plot(lims, lims, color="grey", linestyle=":")
    axes[1, 1].set_title("Q-Q plot: simulated vs historical returns")
    axes[1, 1].set_xlabel("Historical quantiles")
    axes[1, 1].set_ylabel("Simulated quantiles")

    fig.suptitle(f"{ticker} — Realism Diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class MonteCarloPipeline:
    def __init__(self, cfg: MCConfig):
        self.cfg = cfg
        self.loader = MarketDataLoader()
        self.validator = RealismValidator()
        self.backtester = MonteCarloBacktester(cfg)
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.out_dir, "plots").mkdir(exist_ok=True)
        Path(cfg.out_dir, "reports").mkdir(exist_ok=True)
        set_seed(cfg.seed)
        self.model_preds = load_model_predictions(cfg.model_predictions_path)

    def run(self):
        cfg = self.cfg
        ts("=== MONTE CARLO SIMULATION & BACKTESTING START ===")
        ts(f"Engine={cfg.engine} n_sims={cfg.n_sims} horizons={cfg.horizons} "
           f"tickers={list(cfg.tickers)} offline={cfg.offline}")

        if self.model_preds:
            ts(f"Model predictions loaded from '{cfg.model_predictions_path}' for tickers: "
               f"{sorted(self.model_preds.keys())}")
        else:
            ts("No model predictions available — running 100% pure Monte Carlo "
               "(Yahoo Finance data + stochastic simulation) for all tickers/horizons.")

        master_summary = {}
        run_start = time.time()
        for ti, ticker in enumerate(cfg.tickers):
            done = (ti + 1) / len(cfg.tickers)
            ts(f"--- [{ti+1}/{len(cfg.tickers)}] {ticker} | ETA={eta(done, run_start)} ---")
            try:
                result = self._run_ticker(ticker)
                master_summary[ticker] = result
            except Exception as e:
                ts(f"[{ticker}] FAILED: {e}\n{traceback.format_exc()}")
                master_summary[ticker] = {"error": str(e)}

        with open(Path(cfg.out_dir, "master_summary.json"), "w") as fh:
            json.dump(master_summary, fh, indent=2, default=str)
        ts(f"=== COMPLETE — results written to {cfg.out_dir}/ ===")
        self._print_console_summary(master_summary)
        return master_summary

    def _print_console_summary(self, master_summary: Dict):
        """Human-readable recap printed at the end of an interactive run."""
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        for ticker, result in master_summary.items():
            if "error" in result:
                print(f"\n{ticker}: FAILED — {result['error']}")
                continue
            src = "SYNTHETIC (offline fallback)" if result["is_synthetic_data"] else "Yahoo Finance"
            print(f"\n{ticker}  (price source: {src}, current price: {result['current_price']:.2f}, "
                  f"realism score: {result['realism_score']:.0f}/100)")
            for hz in result["horizon_forecast"]:
                tag = "MODEL+MC blend" if hz["model_blended"] else "pure Monte Carlo"
                print(f"  h={hz['horizon_days']:>3d}d [{tag:>16s}] "
                      f"mean={hz['mean_return']:+.2%}  median={hz['median_return']:+.2%}  "
                      f"p(gain)={hz['prob_positive']:.1%}  "
                      f"VaR5={hz.get('VaR_5', float('nan')):+.2%}")
        print("\n" + "=" * 78)
        print(f"Full JSON reports, plots, and per-ticker files are in: {self.cfg.out_dir}/")
        print("=" * 78 + "\n")

    def _run_ticker(self, ticker: str) -> Dict:
        cfg = self.cfg
        rng = np.random.RandomState(abs(hash((ticker, cfg.seed))) % (2 ** 31))

        df, is_synthetic = self.loader.fetch(ticker, cfg.start, cfg.end, cfg.offline, cfg.seed)
        close_full = df["Close"].values.astype(np.float64)
        if len(close_full) < cfg.backtest_min_history + max(cfg.backtest_horizons) + 10:
            ts(f"[{ticker}] WARN: short history ({len(close_full)} rows) — "
               f"backtest may be limited")

        calib_close = close_full[-cfg.calib_lookback_days:] if len(close_full) > cfg.calib_lookback_days \
            else close_full
        log_ret = log_returns_from_close(calib_close)[1:]
        S0 = float(close_full[-1])
        max_h = max(cfg.horizons)

        ts(f"[{ticker}] calibrating '{cfg.engine}' engine on {len(log_ret)} days "
           f"(is_synthetic_data={is_synthetic})")
        engine = build_engine(cfg.engine, log_ret, cfg, rng)

        ts(f"[{ticker}] simulating {cfg.n_sims} paths x {max_h} days")
        paths, sim_log_ret = simulate_price_paths(engine, S0, max_h, cfg.n_sims, rng)

        # ---- realism check (use one representative simulated path per stat) --
        realism = self.validator.validate(ticker, log_ret, sim_log_ret[0])
        # Average realism across a handful of paths for stability
        extra_scores = [self.validator.validate(ticker, log_ret, sim_log_ret[i]).realism_score
                        for i in range(1, min(10, sim_log_ret.shape[0]))]
        realism.realism_score = float(np.mean([realism.realism_score] + extra_scores))
        ts(f"[{ticker}] realism_score={realism.realism_score:.1f}/100 "
           f"(vol {realism.hist_vol_ann:.1%} hist vs {realism.sim_vol_ann:.1%} sim)")

        # ---- multi-horizon forecast summary (+ optional model blend) ---------
        model_hz = self.model_preds.get(ticker, {})
        horizon_summaries = []
        for h in cfg.horizons:
            mp = model_hz.get(h)
            summary = summarize_horizon(paths, S0, h, cfg, model_pred=mp)
            horizon_summaries.append(summary)
            tag = "MODEL+MC" if mp is not None else "PURE MC"
            ts(f"  [{tag}] h={h:>3d}d | mean={summary['mean_return']:+.3%} "
               f"median={summary['median_return']:+.3%} "
               f"p(gain)={summary['prob_positive']:.1%} "
               f"VaR{int(cfg.var_alpha*100)}={summary[f'VaR_{int(cfg.var_alpha*100)}']:+.3%}")

        # ---- rolling-origin backtest of calibration quality -------------------
        ts(f"[{ticker}] running rolling-origin Monte Carlo backtest...")
        backtest_report = self.backtester.run(close_full, cfg.engine, rng)
        if backtest_report.get("per_horizon"):
            for h, stats in backtest_report["per_horizon"].items():
                ts(f"  [Backtest h={h}d] n={stats['n_obs']} "
                   f"PIT_calibrated={stats['pit_calibrated']} "
                   f"VaR_calibrated={stats['var_calibrated']} "
                   f"mean_CRPS={stats['mean_crps']:.5f}")

        # ---- persist artefacts --------------------------------------------------
        report_dir = Path(cfg.out_dir, "reports")
        with open(report_dir / f"{ticker}_horizon_forecast.json", "w") as fh:
            json.dump(horizon_summaries, fh, indent=2)
        with open(report_dir / f"{ticker}_realism.json", "w") as fh:
            json.dump(realism.to_dict(), fh, indent=2)
        with open(report_dir / f"{ticker}_backtest.json", "w") as fh:
            # PIT arrays can be long; keep them in the per-ticker file only
            json.dump(backtest_report, fh, indent=2, default=str)

        plot_dir = Path(cfg.out_dir, "plots")
        hist_tail = close_full[-min(len(close_full), 500):]
        plot_fan_chart(hist_tail, paths, ticker, plot_dir / f"{ticker}_fan_chart.png")
        plot_terminal_distributions(paths, S0, cfg.horizons, ticker,
                                    plot_dir / f"{ticker}_terminal_distributions.png")
        plot_realism_diagnostics(log_ret, sim_log_ret[0], ticker,
                                 plot_dir / f"{ticker}_realism_diagnostics.png")
        if backtest_report.get("per_horizon"):
            plot_pit_calibration(backtest_report, ticker, plot_dir / f"{ticker}_pit_calibration.png")
            plot_coverage_bars(backtest_report, ticker, plot_dir / f"{ticker}_coverage.png")

        return {
            "is_synthetic_data": is_synthetic,
            "current_price": S0,
            "realism_score": realism.realism_score,
            "realism_checks": realism.checks,
            "horizon_forecast": horizon_summaries,
            "backtest_summary": {
                h: {"pit_calibrated": v["pit_calibrated"], "var_calibrated": v["var_calibrated"],
                    "mean_crps": v["mean_crps"], "mean_bias": v["mean_bias"]}
                for h, v in backtest_report.get("per_horizon", {}).items()
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE (user I/O) MODE
# ─────────────────────────────────────────────────────────────────────────────
def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default != "" else ""
    raw = input(f"{msg}{suffix}: ").strip()
    return raw if raw else default


def _prompt_yes_no(msg: str, default_yes: bool = True) -> bool:
    default_str = "Y/n" if default_yes else "y/N"
    raw = input(f"{msg} ({default_str}): ").strip().lower()
    if not raw:
        return default_yes
    return raw.startswith("y")


def _validate_tickers(raw: str) -> Tuple[str, ...]:
    """
    Accepts the exact format we ask the user for: comma-separated ticker
    symbols, e.g.  AAPL, MSFT, GOOG
    Cleans whitespace, upper-cases, drops empties, de-dupes while
    preserving order.
    """
    seen = []
    for chunk in raw.split(","):
        t = chunk.strip().upper()
        if t and t not in seen:
            seen.append(t)
    return tuple(seen)


def run_interactive() -> MCConfig:
    """
    Guided console prompts. Explains the exact input format to the user,
    then decides — per the availability of a model-predictions file —
    whether each ticker will run in MODEL+MC blend mode or PURE Monte
    Carlo (Yahoo Finance data) mode.
    """
    print("=" * 78)
    print("MONTE CARLO RETURN FORECASTER — INTERACTIVE MODE")
    print("=" * 78)
    print(
        "Enter one or more stock tickers, separated by commas.\n"
        "Format example:  AAPL, MSFT, GOOG\n"
    )
    raw_tickers = ""
    tickers: Tuple[str, ...] = ()
    while not tickers:
        raw_tickers = input("Tickers: ").strip()
        tickers = _validate_tickers(raw_tickers)
        if not tickers:
            print("  -> Didn't catch a valid ticker. Please use the format: AAPL, MSFT, GOOG")

    print(f"\nUsing tickers: {', '.join(tickers)}")

    # ---- Detect whether a trained model's predictions are available -------
    print(
        "\nChecking whether a trained forecasting model is currently available...\n"
        "(This looks for a ticker -> {horizon: predicted_return} JSON file,\n"
        f" e.g. one exported by walk_forward_validation.py to "
        f"'{DEFAULT_MODEL_PREDICTIONS_PATH}')"
    )
    custom_path = _prompt(
        "Path to model predictions JSON (leave blank to use the default / skip)",
        default=""
    )
    model_path = custom_path or DEFAULT_MODEL_PREDICTIONS_PATH
    model_preds_probe = load_model_predictions(model_path)

    if model_preds_probe:
        covered = sorted(set(tickers) & set(model_preds_probe.keys()))
        missing = sorted(set(tickers) - set(model_preds_probe.keys()))
        print(f"\n  -> Model predictions FOUND at '{model_path}'.")
        if covered:
            print(f"     Tickers with a model forecast (will run MODEL + Monte Carlo blend): {covered}")
        if missing:
            print(f"     Tickers with NO model forecast (will run PURE Monte Carlo instead): {missing}")
    else:
        print(
            f"\n  -> No usable model predictions found at '{model_path}'.\n"
            "     All tickers will run in PURE MONTE CARLO mode: real price history is\n"
            "     pulled from Yahoo Finance (yfinance) and simulated forward with the\n"
            "     stochastic engine to build the return forecast."
        )
        model_path = None  # nothing to load later

    # ---- Optional overrides (defaults are sane for a quick run) ------------
    print("\nOptional settings — press Enter to accept the default shown.")
    engine = _prompt(
        "Simulation engine (gbm/jump/heston/garch/bootstrap/ensemble)",
        default="ensemble"
    ).lower()
    if engine not in ("gbm", "jump", "heston", "garch", "bootstrap", "ensemble"):
        print(f"  -> '{engine}' not recognized, defaulting to 'ensemble'")
        engine = "ensemble"

    n_sims_raw = _prompt("Number of simulated paths per ticker", default="5000")
    try:
        n_sims = max(int(n_sims_raw), 100)
    except ValueError:
        n_sims = 5000

    horizons_raw = _prompt(
        "Forecast horizons in trading days (comma-separated)",
        default="1,5,10,21,63,126,252"
    )
    try:
        horizons = tuple(int(x.strip()) for x in horizons_raw.split(",") if x.strip())
        if not horizons:
            raise ValueError
    except ValueError:
        horizons = (1, 5, 10, 21, 63, 126, 252)

    start_date = _prompt("Historical data start date (YYYY-MM-DD)", default="2015-01-01")

    force_offline = False
    if not _HAS_YFINANCE:
        print("\n  -> NOTE: yfinance is not installed in this environment, so real market "
              "data can't be pulled. Falling back to the synthetic price generator so "
              "the pipeline still runs end to end.")
        force_offline = True
    else:
        force_offline = not _prompt_yes_no(
            "\nyfinance is available — pull real price history from Yahoo Finance?",
            default_yes=True
        )
        if force_offline:
            print("  -> Using the synthetic data generator instead (offline mode).")

    out_dir = _prompt("Output directory for reports/plots", default="./mc_results")

    cfg = MCConfig(
        tickers=tickers,
        start=start_date,
        out_dir=out_dir,
        engine=engine,
        n_sims=n_sims,
        horizons=horizons,
        model_predictions_path=model_path,
        offline=force_offline,
    )

    print("\n" + "-" * 78)
    print("Starting run with the following configuration:")
    print(f"  Tickers            : {', '.join(cfg.tickers)}")
    print(f"  Data source        : {'synthetic (offline)' if cfg.offline else 'Yahoo Finance (yfinance)'}")
    print(f"  Model predictions  : {cfg.model_predictions_path or 'none — pure Monte Carlo'}")
    print(f"  Engine             : {cfg.engine}")
    print(f"  Simulated paths    : {cfg.n_sims}")
    print(f"  Horizons (days)    : {cfg.horizons}")
    print(f"  Output directory   : {cfg.out_dir}")
    print("-" * 78 + "\n")

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> Optional[MCConfig]:
    """
    Returns None (signalling "go interactive") when the script is invoked
    with no arguments at all, or when --interactive is passed explicitly.
    Otherwise behaves exactly like the original CLI.
    """
    p = argparse.ArgumentParser(description="Monte Carlo backtesting & multi-horizon return simulation")
    p.add_argument("--interactive", action="store_true",
                  help="run the guided console prompts instead of using CLI flags")
    p.add_argument("--tickers", type=str, default=None,
                  help="comma-separated tickers, e.g. AAPL,MSFT,GOOG")
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="./mc_results")
    p.add_argument("--engine", type=str, default="ensemble",
                  choices=["gbm", "jump", "heston", "garch", "bootstrap", "ensemble"])
    p.add_argument("--n-sims", type=int, default=5000)
    p.add_argument("--horizons", type=str, default="1,5,10,21,63,126,252")
    p.add_argument("--block-size", type=int, default=20)
    p.add_argument("--student-t-df", type=float, default=6.0)
    p.add_argument("--risk-free-rate", type=float, default=0.04)
    p.add_argument("--calib-lookback-days", type=int, default=756)
    p.add_argument("--backtest-n-origins", type=int, default=40)
    p.add_argument("--backtest-step", type=int, default=10)
    p.add_argument("--backtest-horizons", type=str, default="5,21,63")
    p.add_argument("--backtest-min-history", type=int, default=504)
    p.add_argument("--backtest-n-sims", type=int, default=1000)
    p.add_argument("--model-predictions", type=str, default=None,
                  help="optional JSON: ticker -> {horizon: predicted_return}. "
                       f"If omitted, defaults to '{DEFAULT_MODEL_PREDICTIONS_PATH}' if it exists.")
    p.add_argument("--model-blend-weight", type=float, default=0.5)
    p.add_argument("--var-alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--offline", action="store_true",
                  help="force synthetic data instead of yfinance (useful for testing)")
    args = p.parse_args()

    # No arguments at all (just `python mc_engine.py`) or explicit --interactive
    # both drop into the guided prompt flow.
    if args.interactive or (len(sys.argv) == 1):
        return None

    if not args.tickers:
        # Still enough info to run, but no tickers given on the CLI —
        # ask for just the tickers rather than bailing out.
        raw = input("Enter comma-separated tickers (e.g. AAPL,MSFT,GOOG): ").strip()
        args.tickers = raw or "AAPL"

    model_path = args.model_predictions
    if model_path is None and Path(DEFAULT_MODEL_PREDICTIONS_PATH).exists():
        model_path = DEFAULT_MODEL_PREDICTIONS_PATH

    return MCConfig(
        tickers=tuple(x.strip().upper() for x in args.tickers.split(",") if x.strip()),
        start=args.start, end=args.end, out_dir=args.out_dir, engine=args.engine,
        n_sims=args.n_sims, horizons=tuple(int(x) for x in args.horizons.split(",")),
        block_size=args.block_size, student_t_df=args.student_t_df,
        risk_free_rate=args.risk_free_rate, calib_lookback_days=args.calib_lookback_days,
        backtest_n_origins=args.backtest_n_origins, backtest_step=args.backtest_step,
        backtest_horizons=tuple(int(x) for x in args.backtest_horizons.split(",")),
        backtest_min_history=args.backtest_min_history, backtest_n_sims=args.backtest_n_sims,
        model_predictions_path=model_path, model_blend_weight=args.model_blend_weight,
        var_alpha=args.var_alpha, seed=args.seed, offline=args.offline,
    )


def main():
    cfg = parse_args()
    if cfg is None:
        cfg = run_interactive()
    if not _HAS_YFINANCE and not cfg.offline:
        ts("NOTE: yfinance not installed — automatically running in offline/synthetic mode. "
           "Install with `pip install yfinance` to use real market data.")
    pipeline = MonteCarloPipeline(cfg)
    pipeline.run()
    ts("=== MONTE CARLO PIPELINE COMPLETE ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)
