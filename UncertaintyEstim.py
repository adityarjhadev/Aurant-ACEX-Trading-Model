"""
Uncertainty Estimation Toolkit
==============================================================================
A standalone module of uncertainty-quantification methods for any point
forecast / regression / classification model, whether it's a classical
statistical model, an ML model, or the output of a Monte Carlo simulation.

"How confident should I be in this prediction?" is answered differently
depending on what you have access to:

  - Only a single point prediction, no model internals?          -> Conformal prediction
  - A model you can retrain many times on resampled data?        -> Bootstrap
  - Multiple diverse models (or seeds) already trained?           -> Ensemble disagreement
  - A model that natively outputs quantiles?                     -> Quantile calibration
  - A parametric residual model (e.g. GARCH)?                     -> Parametric interval propagation
  - Several candidate models you don't want to pick just one of?  -> Bayesian model averaging
  - A stochastic simulator (e.g. Monte Carlo paths)?               -> Distributional summarization

This module implements all of the above as independent, composable
estimators sharing a common `UncertaintyEstimate` result type, plus a
calibration/backtesting suite (PIT histograms, reliability diagrams,
sharpness, pinball loss, CRPS, interval coverage) so any uncertainty
estimator - built here or elsewhere - can be checked against reality
rather than trusted blindly.

Runs standalone
-----------------
`python uncertainty_estimation.py --demo` generates a synthetic noisy
regression problem and demonstrates every estimator end-to-end, including
calibration plots, with no external data or network dependency.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import math
import time
import json
import argparse
import warnings
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import minimize as scipy_minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

EPS = 1e-9
_GLOBAL_START = time.time()


def ts(msg: str):
    e = time.time() - _GLOBAL_START
    h, rem = divmod(int(e), 3600)
    m, s = divmod(rem, 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)


def set_seed(seed: int):
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Shared result container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class UncertaintyEstimate:
    """
    Canonical output of every estimator in this module: a point estimate
    plus a full set of quantiles, so downstream code never needs to know
    which method produced it.
    """
    method: str
    point: np.ndarray                      # [n] point predictions (mean or median)
    quantiles: Dict[float, np.ndarray]     # {q: [n] quantile prediction}, q in (0,1)
    std: Optional[np.ndarray] = None       # [n] std dev, if well-defined
    samples: Optional[np.ndarray] = None   # [n_draws, n] raw samples, if available
    meta: Dict = field(default_factory=dict)

    def interval(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Central (1-alpha) interval: returns (lower, upper) arrays."""
        lo_q, hi_q = alpha / 2, 1 - alpha / 2
        lo = self._nearest_quantile(lo_q)
        hi = self._nearest_quantile(hi_q)
        return lo, hi

    def _nearest_quantile(self, q: float) -> np.ndarray:
        if q in self.quantiles:
            return self.quantiles[q]
        keys = np.array(sorted(self.quantiles.keys()))
        nearest = keys[np.argmin(np.abs(keys - q))]
        return self.quantiles[nearest]

    def to_summary_dict(self, idx: int = 0) -> Dict:
        out = {
            "method": self.method,
            "point": float(self.point[idx]),
            "std": float(self.std[idx]) if self.std is not None else None,
            "quantiles": {str(q): float(v[idx]) for q, v in sorted(self.quantiles.items())},
        }
        return out


def _quantiles_from_samples(samples: np.ndarray, levels: Tuple[float, ...]) -> Dict[float, np.ndarray]:
    """samples: [n_draws, n_points] -> {level: [n_points]}."""
    out = {}
    for q in levels:
        out[q] = np.percentile(samples, q * 100, axis=0)
    return out


DEFAULT_QUANTILE_LEVELS = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BOOTSTRAP UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────
class BootstrapEstimator:
    """
    Resample-and-refit bootstrap. Given a fit function that takes
    (X, y) -> a predict function, this refits the model on B bootstrap
    resamples of the training data and evaluates the resulting predictor
    ensemble on the target points, yielding an empirical distribution of
    predictions that reflects sampling variability in the training data.

    Supports both the standard i.i.d. bootstrap (resample rows with
    replacement) and, for correlated/time-series data, a moving-block
    bootstrap (resample contiguous blocks) so autocorrelation structure
    isn't destroyed.
    """

    def __init__(self, n_boot: int = 500, block_size: int = 1, seed: int = 0):
        self.n_boot = n_boot
        self.block_size = max(block_size, 1)
        self.rng = np.random.RandomState(seed)

    def _resample_indices(self, n: int) -> np.ndarray:
        if self.block_size == 1:
            return self.rng.randint(0, n, size=n)
        idx = np.empty(n, dtype=int)
        filled = 0
        while filled < n:
            start = self.rng.randint(0, n)
            take = min(self.block_size, n - filled)
            block = (start + np.arange(take)) % n
            idx[filled:filled + take] = block
            filled += take
        return idx

    def fit_predict(self, X: np.ndarray, y: np.ndarray, X_target: np.ndarray,
                    fit_fn: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]],
                    quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
                    verbose: bool = False) -> UncertaintyEstimate:
        n = len(y)
        preds = np.empty((self.n_boot, len(X_target)))
        n_failed = 0
        for b in range(self.n_boot):
            idx = self._resample_indices(n)
            Xb, yb = X[idx], y[idx]
            try:
                predict_fn = fit_fn(Xb, yb)
                preds[b] = predict_fn(X_target)
            except Exception:
                preds[b] = np.nan
                n_failed += 1
            if verbose and (b + 1) % max(1, self.n_boot // 10) == 0:
                ts(f"  [Bootstrap] {b+1}/{self.n_boot} resamples fit")
        if n_failed:
            ts(f"  [Bootstrap] WARN: {n_failed}/{self.n_boot} resamples failed to fit")
        valid = ~np.isnan(preds).any(axis=1)
        preds = preds[valid]

        point = np.nanmean(preds, axis=0)
        std = np.nanstd(preds, axis=0, ddof=1)
        quantiles = _quantiles_from_samples(preds, quantile_levels)
        return UncertaintyEstimate(
            method="bootstrap", point=point, quantiles=quantiles, std=std, samples=preds,
            meta={"n_boot_effective": int(valid.sum()), "block_size": self.block_size})


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFORMAL PREDICTION (distribution-free, finite-sample coverage guarantee)
# ─────────────────────────────────────────────────────────────────────────────
class SplitConformalEstimator:
    """
    Split conformal prediction: the model is fit once on a training split;
    residuals on a held-out calibration split are used to build a
    distribution-free prediction interval with a finite-sample marginal
    coverage guarantee (assuming exchangeability): for miscoverage `alpha`,
    P(y in interval) >= 1 - alpha regardless of the underlying model or
    data distribution (unlike bootstrap/Bayesian methods, no distributional
    assumptions are required for this guarantee to hold).
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._q_hat: Optional[float] = None

    def calibrate(self, y_calib: np.ndarray, pred_calib: np.ndarray):
        residuals = np.abs(y_calib - pred_calib)
        n = len(residuals)
        # Finite-sample corrected quantile level (Romano et al. 2019)
        level = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self._q_hat = float(np.quantile(residuals, level))
        return self

    def predict(self, point_pred: np.ndarray,
               quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS) -> UncertaintyEstimate:
        if self._q_hat is None:
            raise RuntimeError("call .calibrate() before .predict()")
        # Under split conformal we only get one guaranteed interval (at
        # `alpha`); we approximate a fuller quantile ladder by scaling the
        # half-width proportionally to a standard-normal z-ratio, purely
        # for downstream API compatibility -- only the `alpha`-level
        # interval carries the formal coverage guarantee.
        z_target = scipy_stats.norm.ppf(1 - self.alpha / 2)
        quantiles = {}
        for q in quantile_levels:
            z_q = scipy_stats.norm.ppf(max(min(q, 0.999), 0.001))
            scale = z_q / z_target if z_target > 0 else 0.0
            quantiles[q] = point_pred + scale * self._q_hat
        return UncertaintyEstimate(
            method="split_conformal", point=point_pred, quantiles=quantiles,
            std=np.full_like(point_pred, self._q_hat / z_target if z_target else 0.0),
            meta={"q_hat": self._q_hat, "alpha": self.alpha,
                 "guaranteed_interval": (self.alpha,)})


class ConformalizedQuantileRegression:
    """
    CQR (Romano, Patterson & Candes 2019): calibrates a pair of lower/upper
    quantile-regression predictions using a held-out calibration set so the
    *width* of the interval adapts locally (heteroscedasticity-aware)
    while still carrying a finite-sample marginal coverage guarantee -
    strictly more informative than split conformal's constant-width band
    when the underlying quantile regressor is reasonable.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._q_hat: Optional[float] = None

    def calibrate(self, y_calib: np.ndarray, lo_calib: np.ndarray, hi_calib: np.ndarray):
        scores = np.maximum(lo_calib - y_calib, y_calib - hi_calib)
        n = len(scores)
        level = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self._q_hat = float(np.quantile(scores, level))
        return self

    def predict(self, lo_pred: np.ndarray, hi_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._q_hat is None:
            raise RuntimeError("call .calibrate() before .predict()")
        return lo_pred - self._q_hat, hi_pred + self._q_hat


# ─────────────────────────────────────────────────────────────────────────────
# 3. QUANTILE REGRESSION (pinball-loss linear quantile regression, no deps)
# ─────────────────────────────────────────────────────────────────────────────
def _pinball_loss(residual: np.ndarray, q: float) -> np.ndarray:
    return np.maximum(q * residual, (q - 1) * residual)


class LinearQuantileRegressor:
    """
    Linear quantile regression fit via subgradient-based convex
    optimization of the pinball ("check") loss, one model per requested
    quantile level. No external ML dependency required. Predictions from
    different quantile levels are sorted post-hoc ("rearrangement", Chernozhukov
    et al. 2010) to guarantee monotonicity (no quantile crossing).
    """

    def __init__(self, quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
                l2_reg: float = 1e-3):
        self.quantile_levels = quantile_levels
        self.l2_reg = l2_reg
        self.coefs_: Dict[float, np.ndarray] = {}

    @staticmethod
    def _design(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.atleast_2d(X)
        if X.shape[0] != len(y):
            X = X.T
        Xd = self._design(X)
        p = Xd.shape[1]
        for q in self.quantile_levels:
            def obj(beta, q=q):
                resid = y - Xd @ beta
                return np.mean(_pinball_loss(resid, q)) + self.l2_reg * np.sum(beta[1:] ** 2)
            beta0 = np.zeros(p)
            beta0[0] = np.quantile(y, q)
            res = scipy_minimize(obj, beta0, method="Nelder-Mead",
                                 options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8})
            self.coefs_[q] = res.x
        return self

    def predict(self, X: np.ndarray) -> UncertaintyEstimate:
        X = np.atleast_2d(X)
        if X.shape[1] != len(next(iter(self.coefs_.values()))) - 1:
            X = X.T
        Xd = self._design(X)
        raw = {q: Xd @ beta for q, beta in self.coefs_.items()}
        sorted_q = sorted(raw.keys())
        stacked = np.stack([raw[q] for q in sorted_q], axis=0)
        stacked_sorted = np.sort(stacked, axis=0)  # rearrangement -> monotone quantiles
        quantiles = {q: stacked_sorted[i] for i, q in enumerate(sorted_q)}
        median = quantiles.get(0.5, stacked_sorted[len(sorted_q) // 2])
        std_proxy = (quantiles.get(0.84, median) - quantiles.get(0.16, median)) / 2.0
        return UncertaintyEstimate(method="quantile_regression", point=median,
                                   quantiles=quantiles, std=np.abs(std_proxy),
                                   meta={"l2_reg": self.l2_reg})


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENSEMBLE DISAGREEMENT (deep-ensemble-style, model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
class EnsembleDisagreementEstimator:
    """
    Given several already-fit models (different architectures, seeds,
    hyperparameters, or bootstrap-trained copies), treats their spread of
    predictions on the same input as an empirical uncertainty distribution
    -- the "deep ensembles" approach (Lakshminarayanan et al. 2017)
    generalized to any model family. Distinguishes *aleatoric* uncertainty
    (average of each member's own predicted variance, if available) from
    *epistemic* uncertainty (variance of the means across members), which
    matters for deciding whether more data or a better model would help.
    """

    def __init__(self, predict_fns: List[Callable[[np.ndarray], np.ndarray]],
                predict_std_fns: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None):
        self.predict_fns = predict_fns
        self.predict_std_fns = predict_std_fns

    def predict(self, X: np.ndarray,
               quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
               n_mc_samples: int = 2000, seed: int = 0) -> UncertaintyEstimate:
        rng = np.random.RandomState(seed)
        member_means = np.stack([fn(X) for fn in self.predict_fns], axis=0)  # [M, n]
        M, n = member_means.shape

        epistemic_var = np.var(member_means, axis=0, ddof=1) if M > 1 else np.zeros(n)
        if self.predict_std_fns:
            member_stds = np.stack([fn(X) for fn in self.predict_std_fns], axis=0)
            aleatoric_var = np.mean(member_stds ** 2, axis=0)
        else:
            aleatoric_var = np.zeros(n)

        total_std = np.sqrt(epistemic_var + aleatoric_var)
        point = np.mean(member_means, axis=0)

        # Build a full predictive sample by mixing: draw a member uniformly,
        # then (if available) add Gaussian noise at that member's aleatoric std.
        samples = np.empty((n_mc_samples, n))
        member_idx = rng.randint(0, M, size=n_mc_samples)
        for s in range(n_mc_samples):
            mi = member_idx[s]
            noise = 0.0
            if self.predict_std_fns:
                noise = rng.normal(0, 1, size=n) * member_stds[mi]
            samples[s] = member_means[mi] + noise

        quantiles = _quantiles_from_samples(samples, quantile_levels)
        return UncertaintyEstimate(
            method="ensemble_disagreement", point=point, quantiles=quantiles,
            std=total_std, samples=samples,
            meta={"n_members": M, "epistemic_std": np.sqrt(epistemic_var).tolist(),
                 "aleatoric_std": np.sqrt(aleatoric_var).tolist()})


# ─────────────────────────────────────────────────────────────────────────────
# 5. BAYESIAN MODEL AVERAGING
# ─────────────────────────────────────────────────────────────────────────────
class BayesianModelAveraging:
    """
    Combines several candidate models into a single predictive distribution
    weighted by each model's approximate posterior probability, estimated
    from an information criterion (BIC by default, which approximates
    -2*log(marginal likelihood) for large n) computed on a training/
    validation set. Rather than picking a single "best" model, BMA
    propagates *model-selection uncertainty* into the final predictive
    interval -- when models disagree substantially, the resulting mixture
    is wider than any individual model's own uncertainty, which is exactly
    the point.
    """

    def __init__(self, model_names: List[str],
                predict_fns: List[Callable[[np.ndarray], np.ndarray]],
                predict_std_fns: List[Callable[[np.ndarray], np.ndarray]],
                log_likelihoods: List[float], n_params: List[int], n_obs: int):
        assert len(model_names) == len(predict_fns) == len(predict_std_fns) == \
               len(log_likelihoods) == len(n_params)
        self.model_names = model_names
        self.predict_fns = predict_fns
        self.predict_std_fns = predict_std_fns
        self.weights = self._bic_weights(log_likelihoods, n_params, n_obs)

    @staticmethod
    def _bic_weights(log_liks: List[float], n_params: List[int], n_obs: int) -> np.ndarray:
        bics = np.array([-2 * ll + k * math.log(max(n_obs, 2)) for ll, k in zip(log_liks, n_params)])
        # Convert BIC differences to approximate posterior model probabilities
        delta = bics - bics.min()
        raw_weights = np.exp(-0.5 * delta)
        return raw_weights / raw_weights.sum()

    def predict(self, X: np.ndarray, n_mc_samples: int = 4000, seed: int = 0,
               quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS) -> UncertaintyEstimate:
        rng = np.random.RandomState(seed)
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        means = np.stack([fn(X) for fn in self.predict_fns], axis=0)     # [M, n]
        stds = np.stack([fn(X) for fn in self.predict_std_fns], axis=0)  # [M, n]

        point = np.tensordot(self.weights, means, axes=(0, 0))

        # Law of total variance: Var = E[within-model var] + Var[model means]
        within = np.tensordot(self.weights, stds ** 2, axes=(0, 0))
        between = np.tensordot(self.weights, (means - point[None, :]) ** 2, axes=(0, 0))
        total_var = within + between

        samples = np.empty((n_mc_samples, n))
        model_idx = rng.choice(len(self.model_names), size=n_mc_samples, p=self.weights)
        for s in range(n_mc_samples):
            mi = model_idx[s]
            samples[s] = means[mi] + rng.normal(0, 1, size=n) * stds[mi]
        quantiles = _quantiles_from_samples(samples, quantile_levels)

        return UncertaintyEstimate(
            method="bayesian_model_averaging", point=point, quantiles=quantiles,
            std=np.sqrt(total_var), samples=samples,
            meta={"model_weights": dict(zip(self.model_names, self.weights.tolist())),
                 "within_model_var_share": float(np.mean(within / (total_var + EPS))),
                 "between_model_var_share": float(np.mean(between / (total_var + EPS)))})


# ─────────────────────────────────────────────────────────────────────────────
# 6. PARAMETRIC / GARCH-STYLE VOLATILITY-PROPAGATED UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────
class GARCHUncertaintyEstimator:
    """
    For a time-varying-volatility residual process (as in a GARCH(1,1)),
    propagates the conditional variance forward analytically to build
    horizon-dependent uncertainty bands -- appropriate when residual
    heteroscedasticity (vol clustering) is the dominant source of
    uncertainty growth with horizon, rather than parameter/model
    uncertainty per se.
    """

    def __init__(self, omega: float, alpha: float, beta: float, nu: float = 8.0):
        self.omega, self.alpha, self.beta, self.nu = omega, alpha, beta, max(nu, 2.1)

    def forecast_variance_path(self, h0_variance: float, last_resid_sq: float,
                               horizon: int) -> np.ndarray:
        """Analytic h-step-ahead conditional variance forecast (no simulation
        needed for GARCH(1,1) because E[eps_t^2] = h_t under correct
        specification)."""
        h = np.empty(horizon)
        h[0] = self.omega + self.alpha * last_resid_sq + self.beta * h0_variance
        long_run = self.omega / max(1 - self.alpha - self.beta, 1e-6)
        persistence = self.alpha + self.beta
        for t in range(1, horizon):
            # Mean-reversion of the multi-step variance forecast to the
            # unconditional (long-run) variance at rate `persistence`.
            h[t] = long_run + (persistence ** t) * (h[0] - long_run)
        return np.maximum(h, 1e-12)

    def cumulative_return_uncertainty(self, mu_daily: float, h0_variance: float,
                                      last_resid_sq: float, horizons: Tuple[int, ...],
                                      quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS
                                      ) -> Dict[int, UncertaintyEstimate]:
        max_h = max(horizons)
        var_path = self.forecast_variance_path(h0_variance, last_resid_sq, max_h)
        cum_var = np.cumsum(var_path)
        t_scale = math.sqrt((self.nu - 2) / self.nu) if self.nu > 2 else 1.0

        out = {}
        for h in horizons:
            cum_mean = mu_daily * h
            cum_std = math.sqrt(cum_var[h - 1]) / t_scale if t_scale > 0 else math.sqrt(cum_var[h - 1])
            quantiles = {q: np.array([cum_mean + scipy_stats.t.ppf(q, df=self.nu) * cum_std])
                        for q in quantile_levels}
            out[h] = UncertaintyEstimate(
                method="garch_analytic", point=np.array([cum_mean]),
                quantiles=quantiles, std=np.array([cum_std]),
                meta={"horizon": h, "student_t_df": self.nu})
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. DISTRIBUTIONAL SUMMARIZATION (for stochastic simulators / MC engines)
# ─────────────────────────────────────────────────────────────────────────────
def summarize_simulation_samples(samples: np.ndarray,
                                 quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS
                                 ) -> UncertaintyEstimate:
    """
    Turns a raw [n_draws, n_points] (or [n_draws]) array of simulator
    output -- e.g. Monte Carlo terminal prices/returns -- into the shared
    UncertaintyEstimate format, computing higher moments so skew/fat-tail
    behaviour isn't silently discarded when only mean+std are reported.
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1 and samples.shape[1] > 1:
        samples = samples.T  # assume caller passed a 1-D array as a row
    point = np.mean(samples, axis=0)
    std = np.std(samples, axis=0, ddof=1)
    skew = scipy_stats.skew(samples, axis=0)
    kurt = scipy_stats.kurtosis(samples, axis=0)
    quantiles = _quantiles_from_samples(samples, quantile_levels)
    return UncertaintyEstimate(
        method="simulation_summary", point=point, quantiles=quantiles, std=std,
        samples=samples, meta={"skew": skew.tolist(), "excess_kurtosis": kurt.tolist(),
                              "n_draws": samples.shape[0]})


# ─────────────────────────────────────────────────────────────────────────────
# Calibration & scoring diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def pit_values(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Probability Integral Transform: for each observation, the fraction of
    simulated/ensemble samples at or below the realized value. If the
    predictive distribution is well calibrated, PIT values are ~Uniform(0,1).
    samples: [n_draws, n_points], y_true: [n_points].
    """
    return np.mean(samples <= y_true[None, :], axis=0)


def pit_uniformity_test(pit: np.ndarray) -> Tuple[float, float]:
    """Kolmogorov-Smirnov test of PIT values against Uniform(0,1)."""
    stat, p = scipy_stats.kstest(pit, "uniform")
    return float(stat), float(p)


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Empirical fraction of true values falling inside [lo, hi]."""
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def pinball_loss_score(y_true: np.ndarray, quantile_pred: np.ndarray, q: float) -> float:
    resid = y_true - quantile_pred
    return float(np.mean(_pinball_loss(resid, q)))


def crps_from_samples(y_true: np.ndarray, samples: np.ndarray) -> float:
    """
    Mean empirical CRPS (energy-score form) across all points:
    CRPS = E|X - y| - 0.5*E|X - X'|. Lower is better; CRPS reduces to MAE
    when the predictive distribution is a point mass, so it penalizes both
    poor location and poor spread.
    """
    n_draws, n = samples.shape
    scores = np.empty(n)
    m = min(n_draws, 400)  # subsample the O(n_draws^2) term for tractability
    for i in range(n):
        s = samples[:, i]
        term1 = np.mean(np.abs(s - y_true[i]))
        sub = np.random.choice(s, size=m, replace=False) if n_draws > m else s
        term2 = 0.5 * np.mean(np.abs(sub[:, None] - sub[None, :]))
        scores[i] = term1 - term2
    return float(np.mean(scores))


def sharpness(lo: np.ndarray, hi: np.ndarray) -> float:
    """Mean interval width -- a well-calibrated but very wide interval is
    still low-value; sharpness quantifies that tradeoff (report alongside
    coverage, never alone)."""
    return float(np.mean(hi - lo))


@dataclass
class CalibrationReport:
    method: str
    nominal_coverage: Dict[float, float]
    empirical_coverage: Dict[float, float]
    mean_pinball: Dict[float, float]
    mean_interval_width: Dict[float, float]
    pit_ks_stat: Optional[float] = None
    pit_ks_pvalue: Optional[float] = None
    mean_crps: Optional[float] = None
    overall_calibrated: bool = False

    def to_dict(self):
        return asdict(self)


class UncertaintyBacktester:
    """
    Checks any `UncertaintyEstimate` against realized outcomes: does the
    advertised (1 - alpha) interval actually contain the true value at
    roughly rate (1 - alpha)? Are quantile predictions well-ranked (low
    pinball loss)? Is the full predictive distribution well-calibrated
    (PIT uniformity, CRPS)? A method that "looks confident" but fails
    these checks is producing misleadingly narrow (or needlessly wide)
    uncertainty and should not be trusted downstream.
    """

    def __init__(self, confidence_levels: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)):
        self.confidence_levels = confidence_levels

    def evaluate(self, estimate: UncertaintyEstimate, y_true: np.ndarray) -> CalibrationReport:
        nominal_cov, empirical_cov, widths = {}, {}, {}
        for cl in self.confidence_levels:
            alpha = 1 - cl
            lo, hi = estimate.interval(alpha)
            empirical_cov[cl] = interval_coverage(y_true, lo, hi)
            nominal_cov[cl] = cl
            widths[cl] = sharpness(lo, hi)

        pinballs = {}
        for q, pred in estimate.quantiles.items():
            pinballs[q] = pinball_loss_score(y_true, pred, q)

        pit_stat = pit_p = crps = None
        if estimate.samples is not None:
            pit = pit_values(y_true, estimate.samples)
            pit_stat, pit_p = pit_uniformity_test(pit)
            crps = crps_from_samples(y_true, estimate.samples)

        # "Calibrated" here means: every tested interval's empirical
        # coverage is within 7 percentage points of nominal, AND (if PIT
        # was computable) the KS test doesn't reject uniformity at 5%.
        cov_ok = all(abs(empirical_cov[cl] - cl) <= 0.07 for cl in self.confidence_levels)
        pit_ok = (pit_p is None) or (pit_p > 0.05)
        overall = bool(cov_ok and pit_ok)

        return CalibrationReport(
            method=estimate.method, nominal_coverage=nominal_cov,
            empirical_coverage=empirical_cov, mean_pinball=pinballs,
            mean_interval_width=widths, pit_ks_stat=pit_stat, pit_ks_pvalue=pit_p,
            mean_crps=crps, overall_calibrated=overall)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_reliability_diagram(report: CalibrationReport, out_path: Path):
    cls = sorted(report.nominal_coverage.keys())
    nominal = [report.nominal_coverage[c] for c in cls]
    empirical = [report.empirical_coverage[c] for c in cls]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey", label="Perfect calibration")
    ax.plot(nominal, empirical, marker="o", color="#4E79A7", label=report.method)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Reliability diagram — {report.method}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pit_histogram(pit: np.ndarray, method: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(pit, bins=20, range=(0, 1), density=True, color="#59A14F", alpha=0.85)
    ax.axhline(1.0, color="grey", linestyle=":")
    ax.set_title(f"PIT histogram — {method} (flat = well-calibrated)")
    ax.set_xlabel("PIT value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_fan(x: np.ndarray, estimate: UncertaintyEstimate, method: str, out_path: Path,
            y_true: Optional[np.ndarray] = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    if 0.05 in estimate.quantiles and 0.95 in estimate.quantiles:
        ax.fill_between(x, estimate.quantiles[0.05], estimate.quantiles[0.95],
                        color="#4E79A7", alpha=0.2, label="90% interval")
    if 0.25 in estimate.quantiles and 0.75 in estimate.quantiles:
        ax.fill_between(x, estimate.quantiles[0.25], estimate.quantiles[0.75],
                        color="#4E79A7", alpha=0.35, label="50% interval")
    ax.plot(x, estimate.point, color="#F28E2B", linewidth=1.8, label="Point estimate")
    if y_true is not None:
        ax.scatter(x, y_true, color="#2c2c2c", s=10, label="Realized", zorder=5)
    ax.set_title(f"Uncertainty fan — {method}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Demo / synthetic harness
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_regression(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """Heteroscedastic nonlinear regression: noise grows with |x|, so a
    well-behaved uncertainty estimator should show *wider* intervals away
    from the origin, not constant-width bands."""
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(-3, 3, size=n))
    true_fn = lambda x: np.sin(x) * 2 + 0.3 * x
    noise_scale = 0.2 + 0.4 * np.abs(x)
    y = true_fn(x) + rng.normal(0, 1, size=n) * noise_scale
    return x, y, true_fn


def _ols_fit_fn(X: np.ndarray, y: np.ndarray) -> Callable:
    Xd = np.column_stack([np.ones(len(X)), X, X ** 2, X ** 3])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    def predict(Xt):
        Xtd = np.column_stack([np.ones(len(Xt)), Xt, Xt ** 2, Xt ** 3])
        return Xtd @ beta
    return predict


def run_demo(out_dir: str, seed: int = 42):
    set_seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts("=== UNCERTAINTY ESTIMATION DEMO START ===")

    x_train, y_train, true_fn = _synthetic_regression(400, seed)
    x_test, y_test, _ = _synthetic_regression(150, seed + 1)

    backtester = UncertaintyBacktester()
    reports = {}

    # 1. Bootstrap ------------------------------------------------------------
    ts("[1/6] Bootstrap resampling...")
    boot = BootstrapEstimator(n_boot=300, seed=seed)
    est_boot = boot.fit_predict(x_train, y_train, x_test, _ols_fit_fn)
    reports["bootstrap"] = backtester.evaluate(est_boot, y_test)
    plot_fan(x_test, est_boot, "bootstrap", Path(out_dir, "bootstrap_fan.png"), y_test)

    # 2. Split conformal -------------------------------------------------------
    ts("[2/6] Split conformal calibration...")
    predict_fn = _ols_fit_fn(x_train, y_train)
    n_cal = len(x_train) // 3
    x_fit, y_fit = x_train[:-n_cal], y_train[:-n_cal]
    x_cal, y_cal = x_train[-n_cal:], y_train[-n_cal:]
    predict_fn2 = _ols_fit_fn(x_fit, y_fit)
    conformal = SplitConformalEstimator(alpha=0.1).calibrate(y_cal, predict_fn2(x_cal))
    est_conf = conformal.predict(predict_fn2(x_test))
    reports["split_conformal"] = backtester.evaluate(est_conf, y_test)
    plot_fan(x_test, est_conf, "split_conformal", Path(out_dir, "conformal_fan.png"), y_test)

    # 3. Quantile regression ---------------------------------------------------
    ts("[3/6] Linear quantile regression (pinball loss, rearranged)...")
    qr = LinearQuantileRegressor(quantile_levels=DEFAULT_QUANTILE_LEVELS).fit(
        np.column_stack([x_train, x_train ** 2]), y_train)
    est_qr = qr.predict(np.column_stack([x_test, x_test ** 2]))
    reports["quantile_regression"] = backtester.evaluate(est_qr, y_test)
    plot_fan(x_test, est_qr, "quantile_regression", Path(out_dir, "qr_fan.png"), y_test)

    # 4. Ensemble disagreement --------------------------------------------------
    ts("[4/6] Ensemble disagreement across bootstrap-trained members...")
    member_fns = []
    rng = np.random.RandomState(seed)
    for _ in range(15):
        idx = rng.randint(0, len(x_train), size=len(x_train))
        member_fns.append(_ols_fit_fn(x_train[idx], y_train[idx]))
    ens = EnsembleDisagreementEstimator(member_fns)
    est_ens = ens.predict(x_test, seed=seed)
    reports["ensemble_disagreement"] = backtester.evaluate(est_ens, y_test)
    plot_fan(x_test, est_ens, "ensemble_disagreement", Path(out_dir, "ensemble_fan.png"), y_test)

    # 5. Bayesian model averaging -----------------------------------------------
    ts("[5/6] Bayesian model averaging over polynomial degrees 1-4...")
    names, pfns, sfns, lls, ks = [], [], [], [], []
    for degree in (1, 2, 3, 4):
        Xd_fit = np.column_stack([x_fit ** d for d in range(1, degree + 1)])
        Xd_fit = np.column_stack([np.ones(len(x_fit)), Xd_fit])
        beta, *_ = np.linalg.lstsq(Xd_fit, y_fit, rcond=None)
        resid = y_fit - Xd_fit @ beta
        sigma2 = np.var(resid)
        ll = float(np.sum(scipy_stats.norm.logpdf(resid, scale=math.sqrt(sigma2))))
        def make_pfn(beta=beta, degree=degree):
            def pfn(Xt):
                Xtd = np.column_stack([Xt ** d for d in range(1, degree + 1)])
                Xtd = np.column_stack([np.ones(len(Xt)), Xtd])
                return Xtd @ beta
            return pfn
        pfn = make_pfn()
        sfn = lambda Xt, s=math.sqrt(sigma2): np.full(len(Xt), s)
        names.append(f"poly_deg{degree}"); pfns.append(pfn); sfns.append(sfn)
        lls.append(ll); ks.append(degree + 1)
    bma = BayesianModelAveraging(names, pfns, sfns, lls, ks, n_obs=len(x_fit))
    est_bma = bma.predict(x_test, seed=seed)
    reports["bayesian_model_averaging"] = backtester.evaluate(est_bma, y_test)
    plot_fan(x_test, est_bma, "bayesian_model_averaging", Path(out_dir, "bma_fan.png"), y_test)
    ts(f"  BMA model weights: {dict(zip(names, np.round(bma.weights, 3).tolist()))}")

    # 6. GARCH-analytic (applied to a synthetic residual vol process) ----------
    ts("[6/6] GARCH-style analytic multi-horizon uncertainty...")
    garch_est = GARCHUncertaintyEstimator(omega=1e-5, alpha=0.08, beta=0.88, nu=7.0)
    horizon_estimates = garch_est.cumulative_return_uncertainty(
        mu_daily=0.0003, h0_variance=2e-4, last_resid_sq=3e-4,
        horizons=(1, 5, 21, 63, 126))
    for h, est in horizon_estimates.items():
        lo, hi = est.interval(0.1)
        ts(f"  h={h:>3d}d | point={est.point[0]:+.3%} 90%-interval=[{lo[0]:+.3%}, {hi[0]:+.3%}]")

    # ---- write out reports ---------------------------------------------------
    all_reports = {k: v.to_dict() for k, v in reports.items()}
    with open(Path(out_dir, "calibration_reports.json"), "w") as fh:
        json.dump(all_reports, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("CALIBRATION SUMMARY (90% intervals)")
    print("=" * 78)
    for name, rep in reports.items():
        emp = rep.empirical_coverage.get(0.9, float("nan"))
        width = rep.mean_interval_width.get(0.9, float("nan"))
        crps = f"{rep.mean_crps:.4f}" if rep.mean_crps is not None else "n/a"
        print(f"{name:>28s} | empirical_cov={emp:.1%}  mean_width={width:.3f}  "
              f"CRPS={crps}  calibrated={rep.overall_calibrated}")
    print("=" * 78)
    print(f"Plots and JSON reports written to: {out_dir}/")
    ts("=== UNCERTAINTY ESTIMATION DEMO COMPLETE ===")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Uncertainty estimation toolkit")
    p.add_argument("--demo", action="store_true", help="run the synthetic end-to-end demo")
    p.add_argument("--out-dir", type=str, default="./uncertainty_results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.demo or len(sys.argv) == 1:
        run_demo(args.out_dir, args.seed)
    else:
        p.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)
