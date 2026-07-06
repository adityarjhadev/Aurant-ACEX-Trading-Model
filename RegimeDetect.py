"""
Regime Detection & Clustering Toolkit
==============================================================================
A standalone module for detecting discrete "regimes" (structurally
different periods) in a time series -- e.g. bull/bear markets, low/high
volatility, calm/crisis -- using several independent, complementary
approaches:

  - Gaussian Hidden Markov Model (EM / Baum-Welch, implemented from
    scratch): learns both the regimes' emission distributions AND their
    transition dynamics (persistence, switching probabilities) jointly.
  - Gaussian Mixture Model clustering: regimes as clusters in a feature
    space, without a temporal transition model (a static baseline / sanity
    check against the HMM).
  - Rolling-window feature clustering (K-means): turns a raw series into a
    rolling feature matrix (mean, vol, skew, autocorrelation, ...) and
    clusters those windows -- useful when "regime" means "a period with a
    certain statistical character" rather than "a certain hidden state".
  - CUSUM change-point detection: flags individual points in time where
    the underlying mean/variance shifted abruptly, independent of assuming
    any fixed number of regimes.
  - Bayesian Online Change Point Detection (BOCPD, Adams & MacKay 2007):
    online, probabilistic run-length posterior -- estimates, at each time
    step, the probability that a new regime started `r` steps ago.
  - Markov-switching regression: like the HMM but for a series with a
    covariate (e.g. regressing returns on a predictor with regime-varying
    coefficients).

Model-selection & validation guardrails
-----------------------------------------
Picking the "right" number of regimes is a genuine problem, not a detail:
this module fits multiple state counts and compares them via BIC/AIC, and
provides regime-persistence diagnostics (implied average regime duration,
transition matrix entropy) so a spuriously over-segmented or trivial
single-state fit doesn't slip through unnoticed. A `RegimeValidator` checks
temporal stability (label switching across bootstrap refits) and reports a
silhouette-style separation score for the discovered regimes.

Runs standalone
-----------------
`python regime_detection.py --demo` generates a synthetic multi-regime
return series (calm / trending / crisis) and runs every detector on it
end to end, comparing recovered regimes against ground truth, with no
external data or network dependency.
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
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.special import logsumexp

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
class RegimeResult:
    method: str
    n_states: int
    labels: np.ndarray                       # [n] hard regime assignment per timestep
    responsibilities: Optional[np.ndarray] = None  # [n, n_states] soft assignment, if available
    state_means: Optional[np.ndarray] = None       # [n_states] or [n_states, n_features]
    state_stds: Optional[np.ndarray] = None
    transition_matrix: Optional[np.ndarray] = None  # [n_states, n_states], if temporal
    log_likelihood: Optional[float] = None
    bic: Optional[float] = None
    aic: Optional[float] = None
    meta: Dict = field(default_factory=dict)

    def regime_durations(self) -> Dict[int, List[int]]:
        """Run-length of each contiguous regime spell, keyed by state id."""
        durations: Dict[int, List[int]] = {s: [] for s in range(self.n_states)}
        if len(self.labels) == 0:
            return durations
        cur, run = self.labels[0], 1
        for lbl in self.labels[1:]:
            if lbl == cur:
                run += 1
            else:
                durations[int(cur)].append(run)
                cur, run = lbl, 1
        durations[int(cur)].append(run)
        return durations

    def average_duration(self) -> Dict[int, float]:
        durs = self.regime_durations()
        return {s: (float(np.mean(v)) if v else 0.0) for s, v in durs.items()}

    def to_summary_dict(self) -> Dict:
        return {
            "method": self.method, "n_states": self.n_states,
            "log_likelihood": self.log_likelihood, "bic": self.bic, "aic": self.aic,
            "average_duration": self.average_duration(),
            "state_means": None if self.state_means is None else np.asarray(self.state_means).tolist(),
            "state_stds": None if self.state_stds is None else np.asarray(self.state_stds).tolist(),
            "transition_matrix": None if self.transition_matrix is None else self.transition_matrix.tolist(),
            "meta": self.meta,
        }


def _bic(log_lik: float, n_params: int, n_obs: int) -> float:
    return -2 * log_lik + n_params * math.log(max(n_obs, 2))


def _aic(log_lik: float, n_params: int) -> float:
    return -2 * log_lik + 2 * n_params


# ─────────────────────────────────────────────────────────────────────────────
# 1. GAUSSIAN HIDDEN MARKOV MODEL (Baum-Welch EM, from scratch)
# ─────────────────────────────────────────────────────────────────────────────
class GaussianHMM:
    """
    A univariate (or diagonal multivariate) Gaussian Hidden Markov Model
    fit by the Baum-Welch EM algorithm, implemented directly with the
    forward-backward algorithm in log-space for numerical stability (no
    `hmmlearn` dependency required). Learns:
      - pi:  [K] initial state distribution
      - A:   [K, K] transition matrix, A[i, j] = P(state_t=j | state_{t-1}=i)
      - mu, sigma: [K] (or [K, D]) emission Gaussian parameters per state

    Regimes are recovered two ways: `predict_states` (Viterbi, the single
    most likely state *sequence*) and `predict_proba` (forward-backward
    posterior, the most likely state *at each time independently* -- these
    can differ, and both are useful).
    """

    def __init__(self, n_states: int = 2, n_iter: int = 200, tol: float = 1e-6,
                seed: int = 0, min_var: float = 1e-6):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.RandomState(seed)
        self.min_var = min_var
        self.pi_: Optional[np.ndarray] = None
        self.A_: Optional[np.ndarray] = None
        self.mu_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.log_likelihood_history_: List[float] = []
        self.converged_: bool = False

    # -- initialization -------------------------------------------------------
    def _init_params(self, x: np.ndarray):
        n = len(x)
        # k-means-style init on quantiles so states start well-separated
        quantile_edges = np.quantile(x, np.linspace(0, 1, self.K + 1))
        self.mu_ = np.array([x[(x >= quantile_edges[k]) & (x <= quantile_edges[k + 1])].mean()
                            if np.any((x >= quantile_edges[k]) & (x <= quantile_edges[k + 1]))
                            else x.mean()
                            for k in range(self.K)])
        self.sigma_ = np.full(self.K, x.std() + EPS)
        self.pi_ = np.full(self.K, 1.0 / self.K)
        # Diagonal-heavy random transition matrix (regimes are persistent by default)
        A = self.rng.uniform(0.01, 0.1, size=(self.K, self.K))
        np.fill_diagonal(A, 1.0 - A.sum(axis=1) + np.diag(A))
        self.A_ = A / A.sum(axis=1, keepdims=True)

    # -- emission log-densities ------------------------------------------------
    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        """Returns [n, K] log N(x_t; mu_k, sigma_k)."""
        n = len(x)
        out = np.empty((n, self.K))
        for k in range(self.K):
            out[:, k] = scipy_stats.norm.logpdf(x, loc=self.mu_[k], scale=self.sigma_[k] + EPS)
        return out

    # -- forward-backward in log space -----------------------------------------
    def _forward_backward(self, log_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        n, K = log_b.shape
        log_A = np.log(self.A_ + EPS)
        log_pi = np.log(self.pi_ + EPS)

        log_alpha = np.empty((n, K))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, n):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]

        log_beta = np.empty((n, K))
        log_beta[-1] = 0.0
        for t in range(n - 2, -1, -1):
            log_beta[t] = logsumexp(log_A + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

        log_lik = logsumexp(log_alpha[-1])
        return log_alpha, log_beta, float(log_lik)

    def fit(self, x: np.ndarray, verbose: bool = False) -> "GaussianHMM":
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        self._init_params(x)
        prev_ll = -np.inf
        self.log_likelihood_history_ = []

        for it in range(self.n_iter):
            log_b = self._log_emission(x)
            log_alpha, log_beta, log_lik = self._forward_backward(log_b)
            self.log_likelihood_history_.append(log_lik)

            # E-step: state posteriors gamma[t,k] and pairwise xi[t,i,j]
            log_gamma = log_alpha + log_beta - log_lik
            gamma = np.exp(log_gamma)
            log_A = np.log(self.A_ + EPS)
            xi_sum = np.zeros((self.K, self.K))
            for t in range(n - 1):
                log_xi_t = (log_alpha[t][:, None] + log_A + log_b[t + 1][None, :]
                           + log_beta[t + 1][None, :] - log_lik)
                xi_sum += np.exp(log_xi_t)

            # M-step
            self.pi_ = gamma[0] / gamma[0].sum()
            denom = xi_sum.sum(axis=1, keepdims=True)
            self.A_ = xi_sum / np.where(denom > EPS, denom, 1.0)
            self.A_ = self.A_ / self.A_.sum(axis=1, keepdims=True)

            w = gamma.sum(axis=0)
            for k in range(self.K):
                if w[k] < EPS:
                    continue
                self.mu_[k] = np.sum(gamma[:, k] * x) / w[k]
                var_k = np.sum(gamma[:, k] * (x - self.mu_[k]) ** 2) / w[k]
                self.sigma_[k] = math.sqrt(max(var_k, self.min_var))

            if verbose and (it + 1) % 20 == 0:
                ts(f"  [HMM K={self.K}] EM iter {it+1}/{self.n_iter} log_lik={log_lik:.3f}")

            if abs(log_lik - prev_ll) < self.tol:
                self.converged_ = True
                if verbose:
                    ts(f"  [HMM K={self.K}] converged at iter {it+1}, log_lik={log_lik:.3f}")
                break
            prev_ll = log_lik

        # Relabel states by ascending mean so state IDs are stable/interpretable
        order = np.argsort(self.mu_)
        self.mu_, self.sigma_, self.pi_ = self.mu_[order], self.sigma_[order], self.pi_[order]
        self.A_ = self.A_[order][:, order]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Forward-backward posterior P(state_t = k | all data), [n, K]."""
        x = np.asarray(x, dtype=np.float64).ravel()
        log_b = self._log_emission(x)
        log_alpha, log_beta, log_lik = self._forward_backward(log_b)
        gamma = np.exp(log_alpha + log_beta - log_lik)
        return gamma / gamma.sum(axis=1, keepdims=True)

    def predict_states(self, x: np.ndarray) -> np.ndarray:
        """Viterbi: single globally-most-likely hidden state sequence."""
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        log_b = self._log_emission(x)
        log_A = np.log(self.A_ + EPS)
        log_pi = np.log(self.pi_ + EPS)

        delta = np.empty((n, self.K))
        psi = np.zeros((n, self.K), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, n):
            scores = delta[t - 1][:, None] + log_A
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_b[t]

        states = np.empty(n, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def score(self, x: np.ndarray) -> float:
        log_b = self._log_emission(np.asarray(x, dtype=np.float64).ravel())
        _, _, log_lik = self._forward_backward(log_b)
        return log_lik

    def n_params(self) -> int:
        # pi (K-1 free) + transition rows (K*(K-1) free) + mu (K) + sigma (K)
        return (self.K - 1) + self.K * (self.K - 1) + self.K + self.K

    def to_regime_result(self, x: np.ndarray, use_viterbi: bool = True) -> RegimeResult:
        proba = self.predict_proba(x)
        labels = self.predict_states(x) if use_viterbi else np.argmax(proba, axis=1)
        ll = self.score(x)
        k = self.n_params()
        n = len(x)
        return RegimeResult(
            method="gaussian_hmm", n_states=self.K, labels=labels, responsibilities=proba,
            state_means=self.mu_.copy(), state_stds=self.sigma_.copy(),
            transition_matrix=self.A_.copy(), log_likelihood=ll,
            bic=_bic(ll, k, n), aic=_aic(ll, k),
            meta={"converged": self.converged_, "decoding": "viterbi" if use_viterbi else "posterior_argmax"})


def select_hmm_states(x: np.ndarray, candidate_k: Tuple[int, ...] = (2, 3, 4),
                      n_iter: int = 200, seed: int = 0, verbose: bool = True
                      ) -> Tuple[GaussianHMM, Dict[int, RegimeResult]]:
    """
    Fits a Gaussian HMM for each candidate state count and selects the one
    minimizing BIC (which penalizes the extra transition/emission
    parameters of larger K more heavily than AIC does -- appropriate here
    since spurious extra regimes are a bigger practical risk than missing
    a real one).
    """
    results = {}
    best_k, best_bic, best_model = None, np.inf, None
    for k in candidate_k:
        model = GaussianHMM(n_states=k, n_iter=n_iter, seed=seed).fit(x, verbose=False)
        result = model.to_regime_result(x)
        results[k] = result
        if verbose:
            ts(f"  [Model selection] K={k}: log_lik={result.log_likelihood:.2f} "
              f"BIC={result.bic:.2f} AIC={result.aic:.2f}")
        if result.bic < best_bic:
            best_k, best_bic, best_model = k, result.bic, model
    if verbose:
        ts(f"  [Model selection] chosen K={best_k} (lowest BIC)")
    return best_model, results


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAUSSIAN MIXTURE MODEL CLUSTERING (static baseline, no transition model)
# ─────────────────────────────────────────────────────────────────────────────
class GaussianMixtureEM:
    """
    Plain (temporally-unaware) Gaussian Mixture Model fit via EM, from
    scratch. Serves as a baseline: if GMM clustering and the HMM recover
    materially different regimes, the *transition* structure the HMM
    captures (persistence, not just where the data points cluster in
    value-space) is doing real work -- worth checking rather than assuming.
    Supports multivariate diagonal-covariance features.
    """

    def __init__(self, n_components: int = 2, n_iter: int = 200, tol: float = 1e-6, seed: int = 0):
        self.K = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.RandomState(seed)
        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.vars_: Optional[np.ndarray] = None
        self.converged_ = False

    def fit(self, X: np.ndarray) -> "GaussianMixtureEM":
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        n, d = X.shape
        # k-means++-style init: pick spread-out starting means
        idx0 = self.rng.randint(n)
        centers = [X[idx0]]
        for _ in range(self.K - 1):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dists / (dists.sum() + EPS)
            centers.append(X[self.rng.choice(n, p=probs)])
        self.means_ = np.array(centers)
        self.vars_ = np.full((self.K, d), X.var(axis=0) + EPS)
        self.weights_ = np.full(self.K, 1.0 / self.K)

        prev_ll = -np.inf
        for it in range(self.n_iter):
            log_resp = np.empty((n, self.K))
            for k in range(self.K):
                log_resp[:, k] = (np.log(self.weights_[k] + EPS)
                                  + np.sum(scipy_stats.norm.logpdf(X, loc=self.means_[k],
                                                                   scale=np.sqrt(self.vars_[k]) + EPS), axis=1))
            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp - log_norm)
            ll = float(np.sum(log_norm))

            Nk = resp.sum(axis=0) + EPS
            self.weights_ = Nk / n
            self.means_ = (resp.T @ X) / Nk[:, None]
            for k in range(self.K):
                diff = X - self.means_[k]
                self.vars_[k] = np.maximum((resp[:, k][:, None] * diff ** 2).sum(axis=0) / Nk[k], EPS)

            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                break
            prev_ll = ll
        self._last_ll = prev_ll
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and self.means_.shape[1] > 1:
            X = X.T
        n = X.shape[0]
        log_resp = np.empty((n, self.K))
        for k in range(self.K):
            log_resp[:, k] = (np.log(self.weights_[k] + EPS)
                              + np.sum(scipy_stats.norm.logpdf(X, loc=self.means_[k],
                                                               scale=np.sqrt(self.vars_[k]) + EPS), axis=1))
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        return np.exp(log_resp - log_norm)

    def n_params(self, d: int) -> int:
        return (self.K - 1) + self.K * d + self.K * d  # weights + means + diag vars

    def to_regime_result(self, X: np.ndarray) -> RegimeResult:
        X2 = np.atleast_2d(X)
        if X2.shape[0] == 1 and self.means_.shape[1] > 1:
            X2 = X2.T
        proba = self.predict_proba(X2)
        labels = np.argmax(proba, axis=1)
        n, d = X2.shape
        k = self.n_params(d)
        return RegimeResult(
            method="gmm_clustering", n_states=self.K, labels=labels, responsibilities=proba,
            state_means=self.means_.copy(), state_stds=np.sqrt(self.vars_).copy(),
            log_likelihood=self._last_ll, bic=_bic(self._last_ll, k, n), aic=_aic(self._last_ll, k),
            meta={"converged": self.converged_})


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLLING-WINDOW FEATURE CLUSTERING (K-means on statistical fingerprints)
# ─────────────────────────────────────────────────────────────────────────────
def rolling_feature_matrix(x: np.ndarray, window: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a [n - window + 1, n_features] matrix of rolling statistical
    fingerprints (mean, vol, skew, kurtosis, lag-1 autocorrelation, and
    the fraction of down-moves) so "regime" can mean "a period with this
    statistical character" rather than "a period with this hidden state".
    Returns (features, end_indices) where end_indices[i] is the original
    series index the i-th window ends at (for re-aligning labels to time).
    """
    n = len(x)
    n_windows = n - window + 1
    feats = np.empty((n_windows, 6))
    end_idx = np.empty(n_windows, dtype=int)
    for i in range(n_windows):
        w = x[i:i + window]
        feats[i, 0] = np.mean(w)
        feats[i, 1] = np.std(w, ddof=1)
        feats[i, 2] = scipy_stats.skew(w)
        feats[i, 3] = scipy_stats.kurtosis(w)
        if len(w) > 2:
            feats[i, 4] = np.corrcoef(w[:-1], w[1:])[0, 1] if np.std(w[:-1]) > EPS else 0.0
        else:
            feats[i, 4] = 0.0
        feats[i, 5] = np.mean(w < 0)
        end_idx[i] = i + window - 1
    feats = np.nan_to_num(feats, nan=0.0)
    return feats, end_idx


class KMeansScratch:
    """Lightweight K-means (k-means++ init, Lloyd's algorithm) so rolling-
    feature clustering has no external ML dependency."""

    def __init__(self, n_clusters: int = 3, n_iter: int = 300, n_init: int = 10, seed: int = 0):
        self.K = n_clusters
        self.n_iter = n_iter
        self.n_init = n_init
        self.rng = np.random.RandomState(seed)
        self.centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    def _kpp_init(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        centers = [X[self.rng.randint(n)]]
        for _ in range(self.K - 1):
            d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = d2 / (d2.sum() + EPS)
            centers.append(X[self.rng.choice(n, p=probs)])
        return np.array(centers)

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        X = np.asarray(X, dtype=np.float64)
        # standardize features so scale differences don't dominate the metric
        self._mu, self._sd = X.mean(axis=0), X.std(axis=0) + EPS
        Xs = (X - self._mu) / self._sd

        best_inertia, best_centers, best_labels = np.inf, None, None
        for _ in range(self.n_init):
            centers = self._kpp_init(Xs)
            labels = np.zeros(len(Xs), dtype=int)
            for _ in range(self.n_iter):
                dists = np.stack([np.sum((Xs - c) ** 2, axis=1) for c in centers], axis=1)
                new_labels = np.argmin(dists, axis=1)
                if np.array_equal(new_labels, labels) and _ > 0:
                    break
                labels = new_labels
                for k in range(self.K):
                    if np.any(labels == k):
                        centers[k] = Xs[labels == k].mean(axis=0)
            inertia = float(np.sum((Xs - centers[labels]) ** 2))
            if inertia < best_inertia:
                best_inertia, best_centers, best_labels = inertia, centers, labels
        self.centers_ = best_centers
        self.inertia_ = best_inertia
        self._labels = best_labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self._mu) / self._sd
        dists = np.stack([np.sum((Xs - c) ** 2, axis=1) for c in self.centers_], axis=1)
        return np.argmin(dists, axis=1)

    def silhouette_score(self, X: np.ndarray, labels: np.ndarray, sample_cap: int = 500) -> float:
        """Mean silhouette coefficient (computed on a subsample if large,
        since the naive O(n^2) pairwise-distance version doesn't scale)."""
        Xs = (np.asarray(X, dtype=np.float64) - self._mu) / self._sd
        n = len(Xs)
        if n > sample_cap:
            idx = self.rng.choice(n, size=sample_cap, replace=False)
            Xs, labels = Xs[idx], labels[idx]
            n = sample_cap
        if len(set(labels.tolist())) < 2:
            return 0.0
        dmat = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2))
        sil = np.zeros(n)
        for i in range(n):
            same = labels == labels[i]
            same[i] = False
            a = dmat[i, same].mean() if same.any() else 0.0
            b = np.inf
            for k in set(labels.tolist()):
                if k == labels[i]:
                    continue
                other = labels == k
                if other.any():
                    b = min(b, dmat[i, other].mean())
            sil[i] = 0.0 if max(a, b) < EPS else (b - a) / max(a, b)
        return float(np.mean(sil))


def cluster_rolling_regimes(x: np.ndarray, window: int = 21, n_clusters: int = 3,
                            seed: int = 0) -> Tuple[RegimeResult, np.ndarray]:
    """
    Full pipeline: rolling feature extraction -> K-means clustering ->
    re-aligned per-timestep labels (each raw timestep inherits the label of
    the most recent window ending at or before it). Returns (RegimeResult
    over the *original* series length, the rolling feature matrix used).
    """
    feats, end_idx = rolling_feature_matrix(x, window)
    km = KMeansScratch(n_clusters=n_clusters, seed=seed).fit(feats)
    window_labels = km.predict(feats)
    sil = km.silhouette_score(feats, window_labels)

    labels = np.full(len(x), -1, dtype=int)
    labels[end_idx] = window_labels
    # Forward-fill so every original timestep has a label (windows earlier
    # than `window` inherit the first available window's label)
    last = window_labels[0]
    for i in range(len(x)):
        if labels[i] == -1:
            labels[i] = last
        else:
            last = labels[i]

    state_means = np.array([feats[window_labels == k, 0].mean() if np.any(window_labels == k) else np.nan
                            for k in range(n_clusters)])
    state_stds = np.array([feats[window_labels == k, 1].mean() if np.any(window_labels == k) else np.nan
                           for k in range(n_clusters)])

    result = RegimeResult(
        method="rolling_kmeans", n_states=n_clusters, labels=labels,
        state_means=state_means, state_stds=state_stds,
        meta={"window": window, "silhouette_score": sil, "inertia": km.inertia_,
             "feature_names": ["mean", "vol", "skew", "kurtosis", "autocorr_lag1", "frac_down"]})
    return result, feats


# ─────────────────────────────────────────────────────────────────────────────
# 4. CUSUM CHANGE-POINT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def cusum_changepoints(x: np.ndarray, threshold: float = 5.0, drift: float = 0.0) -> List[int]:
    """
    Two-sided CUSUM (cumulative sum control chart): flags a change-point
    whenever the cumulative deviation from the running mean (net of an
    allowed `drift`) exceeds `threshold` standard deviations, then resets.
    Simple, fast, and a reasonable "first pass" complement to the
    model-based methods above -- it doesn't require choosing a number of
    regimes ahead of time, only a sensitivity threshold.
    """
    x = np.asarray(x, dtype=np.float64)
    mean, std = np.mean(x), np.std(x, ddof=1) + EPS
    z = (x - mean) / std
    pos, neg = 0.0, 0.0
    changepoints = []
    for t in range(len(z)):
        pos = max(0.0, pos + z[t] - drift)
        neg = min(0.0, neg + z[t] + drift)
        if pos > threshold:
            changepoints.append(t)
            pos, neg = 0.0, 0.0
        elif neg < -threshold:
            changepoints.append(t)
            pos, neg = 0.0, 0.0
    return changepoints


def segments_from_changepoints(n: int, changepoints: List[int]) -> np.ndarray:
    """Converts a change-point list into a per-timestep segment-id label array."""
    labels = np.zeros(n, dtype=int)
    seg = 0
    cps = sorted(changepoints)
    cp_iter = iter(cps)
    next_cp = next(cp_iter, None)
    for t in range(n):
        if next_cp is not None and t >= next_cp:
            seg += 1
            next_cp = next(cp_iter, None)
        labels[t] = seg
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 5. BAYESIAN ONLINE CHANGE POINT DETECTION (BOCPD, Adams & MacKay 2007)
# ─────────────────────────────────────────────────────────────────────────────
class BOCPD:
    """
    Online run-length posterior for a Gaussian observation model with a
    Normal-Inverse-Gamma conjugate prior, using a constant hazard function
    (geometric prior on regime duration with mean `1/hazard`). At each
    time step this maintains the full posterior P(run_length_t = r | data)
    which is exactly the probability that the current regime started `r`
    steps ago -- distinct from CUSUM/HMM in giving a probabilistic,
    streaming (no look-ahead) answer to "did a change just happen?"
    """

    def __init__(self, hazard: float = 1.0 / 100, mu0: float = 0.0, kappa0: float = 1.0,
                alpha0: float = 1.0, beta0: float = 1.0):
        self.hazard = hazard
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0

    def run(self, x: np.ndarray, max_run_length: Optional[int] = None) -> Dict[str, np.ndarray]:
        n = len(x)
        max_run_length = max_run_length or n
        # R[t, r] = P(run length = r at time t | x_1..x_t)
        R = np.zeros((n + 1, max_run_length + 1))
        R[0, 0] = 1.0

        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])

        most_likely_run_length = np.zeros(n, dtype=int)
        changepoint_prob = np.zeros(n)

        for t in range(n):
            xt = x[t]
            # Predictive prob under a Student-t (from the NIG posterior) for each active run length
            df = 2 * alpha
            scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
            pred_probs = scipy_stats.t.pdf(xt, df=df, loc=mu, scale=scale + EPS)

            r_max = min(len(pred_probs), max_run_length)
            pred_probs = pred_probs[:r_max]
            growth = R[t, :r_max] * pred_probs * (1 - self.hazard)
            cp_mass = np.sum(R[t, :r_max] * pred_probs * self.hazard)

            new_R = np.zeros(max_run_length + 1)
            new_R[1:r_max + 1] = growth
            new_R[0] = cp_mass
            total = new_R.sum()
            if total > EPS:
                new_R /= total
            R[t + 1] = new_R

            changepoint_prob[t] = new_R[0]
            most_likely_run_length[t] = int(np.argmax(new_R))

            # Update sufficient statistics for each hypothetical run length
            new_kappa = np.concatenate([[self.kappa0], kappa + 1])
            new_mu = np.concatenate([[self.mu0], (kappa * mu + xt) / (kappa + 1)])
            new_alpha = np.concatenate([[self.alpha0], alpha + 0.5])
            new_beta = np.concatenate([[self.beta0],
                                       beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))])
            mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta
            # Cap the growing state vectors at max_run_length for tractability
            if len(mu) > max_run_length + 1:
                mu, kappa, alpha, beta = mu[:max_run_length + 1], kappa[:max_run_length + 1], \
                                         alpha[:max_run_length + 1], beta[:max_run_length + 1]

        changepoints = [t for t in range(1, n) if changepoint_prob[t] > 0.5]
        return {
            "run_length_posterior": R[1:], "changepoint_prob": changepoint_prob,
            "most_likely_run_length": most_likely_run_length, "changepoints": changepoints,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. MARKOV-SWITCHING REGRESSION (regime-dependent regression coefficients)
# ─────────────────────────────────────────────────────────────────────────────
class MarkovSwitchingRegression:
    """
    Extends the Gaussian HMM to a series with a covariate: y_t = beta_k *
    x_t + c_k + eps_t, where the regression coefficients (and residual
    variance) depend on the current hidden regime k. Fit via EM exactly
    like GaussianHMM, but the M-step for the emission parameters is a
    per-state weighted least squares regression instead of a weighted mean.
    Useful when "regime" means "a period where the relationship between
    two series changed" (e.g. a hedge ratio or market beta that shifts
    across crisis vs calm periods) rather than just a level/vol shift in a
    single series.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 200, tol: float = 1e-6, seed: int = 0):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.RandomState(seed)
        self.pi_ = self.A_ = self.beta_ = self.sigma_ = None

    def fit(self, y: np.ndarray, x: np.ndarray, verbose: bool = False) -> "MarkovSwitchingRegression":
        y = np.asarray(y, dtype=np.float64).ravel()
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(y)
        Xd = np.column_stack([np.ones(n), x])

        # init: split data into K quantile buckets of x, run OLS per bucket
        edges = np.quantile(x, np.linspace(0, 1, self.K + 1))
        self.beta_ = np.zeros((self.K, 2))
        self.sigma_ = np.ones(self.K)
        for k in range(self.K):
            mask = (x >= edges[k]) & (x <= edges[k + 1])
            if mask.sum() > 2:
                b, *_ = np.linalg.lstsq(Xd[mask], y[mask], rcond=None)
                self.beta_[k] = b
                resid = y[mask] - Xd[mask] @ b
                self.sigma_[k] = max(resid.std(), EPS)
            else:
                b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
                self.beta_[k] = b + self.rng.normal(0, 0.1, size=2)
                self.sigma_[k] = y.std()

        self.pi_ = np.full(self.K, 1.0 / self.K)
        A = self.rng.uniform(0.01, 0.1, size=(self.K, self.K))
        np.fill_diagonal(A, 1.0 - A.sum(axis=1) + np.diag(A))
        self.A_ = A / A.sum(axis=1, keepdims=True)

        prev_ll = -np.inf
        for it in range(self.n_iter):
            log_b = np.empty((n, self.K))
            for k in range(self.K):
                mu_k = Xd @ self.beta_[k]
                log_b[:, k] = scipy_stats.norm.logpdf(y, loc=mu_k, scale=self.sigma_[k] + EPS)

            log_alpha, log_beta, log_lik = self._forward_backward(log_b)
            log_gamma = log_alpha + log_beta - log_lik
            gamma = np.exp(log_gamma)

            log_A = np.log(self.A_ + EPS)
            xi_sum = np.zeros((self.K, self.K))
            for t in range(n - 1):
                log_xi_t = (log_alpha[t][:, None] + log_A + log_b[t + 1][None, :]
                           + log_beta[t + 1][None, :] - log_lik)
                xi_sum += np.exp(log_xi_t)

            self.pi_ = gamma[0] / gamma[0].sum()
            denom = xi_sum.sum(axis=1, keepdims=True)
            self.A_ = xi_sum / np.where(denom > EPS, denom, 1.0)
            self.A_ = self.A_ / self.A_.sum(axis=1, keepdims=True)

            for k in range(self.K):
                w = gamma[:, k]
                if w.sum() < EPS:
                    continue
                W = np.diag(w)
                XtWX = Xd.T @ W @ Xd + np.eye(2) * 1e-8
                XtWy = Xd.T @ W @ y
                self.beta_[k] = np.linalg.solve(XtWX, XtWy)
                resid = y - Xd @ self.beta_[k]
                var_k = np.sum(w * resid ** 2) / w.sum()
                self.sigma_[k] = math.sqrt(max(var_k, 1e-8))

            if verbose and (it + 1) % 20 == 0:
                ts(f"  [MS-Regression K={self.K}] iter {it+1} log_lik={log_lik:.3f}")
            if abs(log_lik - prev_ll) < self.tol:
                break
            prev_ll = log_lik

        self._last_ll = prev_ll
        # Relabel by ascending slope coefficient for interpretability
        order = np.argsort(self.beta_[:, 1])
        self.beta_, self.sigma_, self.pi_ = self.beta_[order], self.sigma_[order], self.pi_[order]
        self.A_ = self.A_[order][:, order]
        return self

    def _forward_backward(self, log_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        n, K = log_b.shape
        log_A = np.log(self.A_ + EPS)
        log_pi = np.log(self.pi_ + EPS)
        log_alpha = np.empty((n, K))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, n):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]
        log_beta = np.empty((n, K))
        log_beta[-1] = 0.0
        for t in range(n - 2, -1, -1):
            log_beta[t] = logsumexp(log_A + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
        log_lik = logsumexp(log_alpha[-1])
        return log_alpha, log_beta, float(log_lik)

    def predict_states(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = len(y)
        Xd = np.column_stack([np.ones(n), np.asarray(x).ravel()])
        log_b = np.empty((n, self.K))
        for k in range(self.K):
            mu_k = Xd @ self.beta_[k]
            log_b[:, k] = scipy_stats.norm.logpdf(y, loc=mu_k, scale=self.sigma_[k] + EPS)
        log_alpha, log_beta, log_lik = self._forward_backward(log_b)
        gamma = np.exp(log_alpha + log_beta - log_lik)
        return np.argmax(gamma, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Regime validation / stability diagnostics
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RegimeValidationReport:
    method: str
    n_states: int
    average_duration: Dict[int, float]
    transition_entropy: Optional[float]     # mean row entropy of transition matrix (lower = more persistent)
    label_stability_ari: Optional[float]    # adjusted-Rand-like agreement across bootstrap refits
    separation_score: Optional[float]       # silhouette-style separation of state emission distributions
    degenerate_state_warning: bool          # True if any state captures too few observations to trust

    def to_dict(self):
        return asdict(self)


def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Standard ARI computed from a contingency table -- used here to check
    whether regime labels are stable across bootstrap refits (1.0 = perfect
    agreement up to relabeling, ~0 = no better than chance)."""
    a_ids, b_ids = np.unique(labels_a), np.unique(labels_b)
    contingency = np.zeros((len(a_ids), len(b_ids)))
    for i, ai in enumerate(a_ids):
        for j, bj in enumerate(b_ids):
            contingency[i, j] = np.sum((labels_a == ai) & (labels_b == bj))
    n = len(labels_a)
    sum_comb_c = np.sum([math.comb(int(v), 2) for v in contingency.sum(axis=1)])
    sum_comb_k = np.sum([math.comb(int(v), 2) for v in contingency.sum(axis=0)])
    sum_comb = np.sum([math.comb(int(v), 2) for v in contingency.ravel()])
    expected = sum_comb_c * sum_comb_k / max(math.comb(n, 2), 1)
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    denom = max_index - expected
    if abs(denom) < EPS:
        return 1.0
    return float((sum_comb - expected) / denom)


def _transition_entropy(A: np.ndarray) -> float:
    row_entropy = -np.sum(A * np.log(A + EPS), axis=1)
    return float(np.mean(row_entropy))


def validate_hmm_regimes(x: np.ndarray, n_states: int, n_bootstrap: int = 8, seed: int = 0,
                         min_state_frac: float = 0.03, n_iter: int = 60) -> RegimeValidationReport:
    """
    Refits the HMM on `n_bootstrap` block-bootstrap resamples of the same
    series and measures how consistently the same regime labels come back
    (via ARI against the full-sample fit), plus checks each state actually
    captures a non-trivial share of observations (a state with <3% of the
    data is more likely a fitting artifact than a real regime).
    """
    full_model = GaussianHMM(n_states=n_states, seed=seed, n_iter=n_iter).fit(x)
    full_result = full_model.to_regime_result(x)

    rng = np.random.RandomState(seed)
    n = len(x)
    block = max(n // 20, 5)
    aris = []
    for b in range(n_bootstrap):
        idx = np.empty(n, dtype=int)
        filled = 0
        while filled < n:
            start = rng.randint(0, n)
            take = min(block, n - filled)
            idx[filled:filled + take] = (start + np.arange(take)) % n
            filled += take
        xb = x[idx]
        try:
            model_b = GaussianHMM(n_states=n_states, seed=seed + b + 1, n_iter=n_iter).fit(xb)
            labels_b = model_b.to_regime_result(xb).labels
            aris.append(_adjusted_rand_index(full_result.labels[idx], labels_b))
        except Exception:
            continue

    state_fracs = np.array([np.mean(full_result.labels == k) for k in range(n_states)])
    degenerate = bool(np.any(state_fracs < min_state_frac))

    # Separation score: ratio of between-state mean distance to average within-state std
    between = np.std(full_model.mu_) if n_states > 1 else 0.0
    within = np.mean(full_model.sigma_)
    separation = float(between / (within + EPS))

    return RegimeValidationReport(
        method="gaussian_hmm", n_states=n_states,
        average_duration=full_result.average_duration(),
        transition_entropy=_transition_entropy(full_model.A_),
        label_stability_ari=float(np.mean(aris)) if aris else None,
        separation_score=separation, degenerate_state_warning=degenerate)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
_REGIME_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#59A14F", "#B07AA1", "#76B7B2"]


def plot_regime_overlay(x: np.ndarray, labels: np.ndarray, title: str, out_path: Path,
                        true_labels: Optional[np.ndarray] = None):
    n_rows = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 4.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)
    t_idx = np.arange(len(x))

    ax = axes[0]
    for k in sorted(set(labels.tolist())):
        mask = labels == k
        ax.scatter(t_idx[mask], x[mask], s=6, color=_REGIME_COLORS[k % len(_REGIME_COLORS)],
                  label=f"regime {k}")
    ax.plot(t_idx, x, color="grey", alpha=0.25, linewidth=0.6, zorder=0)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)

    if true_labels is not None:
        ax = axes[1]
        for k in sorted(set(true_labels.tolist())):
            mask = true_labels == k
            ax.scatter(t_idx[mask], x[mask], s=6, color=_REGIME_COLORS[k % len(_REGIME_COLORS)],
                      label=f"true regime {k}")
        ax.plot(t_idx, x, color="grey", alpha=0.25, linewidth=0.6, zorder=0)
        ax.set_title("Ground truth regimes")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_transition_matrix(A: np.ndarray, out_path: Path, title: str = "Transition matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                   color="white" if A[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(A.shape[1])); ax.set_yticks(range(A.shape[0]))
    ax.set_xlabel("to state"); ax.set_ylabel("from state")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_bocpd(x: np.ndarray, bocpd_out: Dict, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(x, color="#2c2c2c", linewidth=0.8)
    for cp in bocpd_out["changepoints"]:
        axes[0].axvline(cp, color="#E15759", linestyle=":", alpha=0.7)
    axes[0].set_title("Series with detected BOCPD change-points")
    axes[1].plot(bocpd_out["changepoint_prob"], color="#4E79A7")
    axes[1].set_title("P(change-point at t)")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic multi-regime demo harness
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_regime_series(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a 3-regime synthetic return series -- calm (low vol, small
    positive drift), trending (moderate vol, strong positive drift), and
    crisis (high vol, negative drift) -- with realistic persistence
    (regimes last tens of days, not single ticks) via an explicit Markov
    chain, so recovered regimes can be checked against a known truth.
    """
    rng = np.random.RandomState(seed)
    regime_params = {
        0: (0.0002, 0.006),   # calm
        1: (0.0012, 0.012),   # trending
        2: (-0.0020, 0.028),  # crisis
    }
    A = np.array([
        [0.97, 0.02, 0.01],
        [0.03, 0.95, 0.02],
        [0.05, 0.05, 0.90],
    ])
    state = 0
    states = np.empty(n, dtype=int)
    x = np.empty(n)
    for t in range(n):
        states[t] = state
        mu, sigma = regime_params[state]
        x[t] = rng.normal(mu, sigma)
        state = rng.choice(3, p=A[state])
    return x, states


def run_demo(out_dir: str, seed: int = 7):
    set_seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts("=== REGIME DETECTION & CLUSTERING DEMO START ===")

    x, true_states = _synthetic_regime_series(800, seed)
    ts(f"Generated synthetic 3-regime series, n={len(x)}")

    # 1. HMM with model selection over K -----------------------------------
    ts("[1/6] Gaussian HMM with BIC-based state-count selection...")
    best_model, selection_results = select_hmm_states(x, candidate_k=(2, 3, 4), seed=seed, n_iter=80)
    hmm_result = best_model.to_regime_result(x)
    ari_vs_truth = _adjusted_rand_index(true_states, hmm_result.labels)
    ts(f"  Selected K={best_model.K}, ARI vs ground truth = {ari_vs_truth:.3f}")
    plot_regime_overlay(x, hmm_result.labels, f"Gaussian HMM regimes (K={best_model.K})",
                        Path(out_dir, "hmm_regimes.png"), true_labels=true_states)
    plot_transition_matrix(hmm_result.transition_matrix, Path(out_dir, "hmm_transition_matrix.png"))

    # 2. Validation / stability diagnostics ---------------------------------
    ts("[2/6] Bootstrap stability & separation diagnostics...")
    validation = validate_hmm_regimes(x, n_states=best_model.K, n_bootstrap=6, seed=seed, n_iter=50)
    ts(f"  label_stability_ARI={validation.label_stability_ari:.3f} "
      f"separation_score={validation.separation_score:.2f} "
      f"transition_entropy={validation.transition_entropy:.3f} "
      f"degenerate_state_warning={validation.degenerate_state_warning}")

    # 3. GMM static baseline ---------------------------------------------------
    ts("[3/6] Gaussian Mixture Model (static, no transition model)...")
    gmm = GaussianMixtureEM(n_components=best_model.K, seed=seed).fit(x.reshape(-1, 1))
    gmm_result = gmm.to_regime_result(x.reshape(-1, 1))
    ari_gmm_vs_hmm = _adjusted_rand_index(hmm_result.labels, gmm_result.labels)
    ts(f"  GMM vs HMM label agreement (ARI)={ari_gmm_vs_hmm:.3f} "
      "(low agreement => transition persistence is doing real work)")
    plot_regime_overlay(x, gmm_result.labels, "GMM static clustering",
                        Path(out_dir, "gmm_regimes.png"), true_labels=true_states)

    # 4. Rolling-window feature clustering ------------------------------------
    ts("[4/6] Rolling-window statistical fingerprint clustering (K-means)...")
    rolling_result, feats = cluster_rolling_regimes(x, window=21, n_clusters=best_model.K, seed=seed)
    ts(f"  silhouette_score={rolling_result.meta['silhouette_score']:.3f} "
      f"inertia={rolling_result.meta['inertia']:.1f}")
    plot_regime_overlay(x, rolling_result.labels, "Rolling-window feature K-means",
                        Path(out_dir, "rolling_kmeans_regimes.png"), true_labels=true_states)

    # 5. CUSUM + BOCPD change-point detection -----------------------------------
    ts("[5/6] CUSUM and Bayesian Online Change Point Detection...")
    cps = cusum_changepoints(x, threshold=6.0)
    ts(f"  CUSUM flagged {len(cps)} change-points")
    seg_labels = segments_from_changepoints(len(x), cps)
    plot_regime_overlay(x, seg_labels % len(_REGIME_COLORS), "CUSUM segments",
                        Path(out_dir, "cusum_segments.png"), true_labels=true_states)

    bocpd = BOCPD(hazard=1.0 / 150)
    bocpd_out = bocpd.run(x, max_run_length=300)
    ts(f"  BOCPD flagged {len(bocpd_out['changepoints'])} change-points "
      f"(P(changepoint) > 0.5)")
    plot_bocpd(x, bocpd_out, Path(out_dir, "bocpd.png"))

    # 6. Markov-switching regression (regime-varying beta) ----------------------
    ts("[6/6] Markov-switching regression on a synthetic regime-varying beta...")
    rng = np.random.RandomState(seed + 1)
    factor = rng.normal(0, 1, size=len(x))
    beta_true = np.where(true_states == 2, 1.8, 0.4)  # beta jumps up in "crisis"
    y = beta_true * factor + rng.normal(0, 0.3, size=len(x))
    ms_reg = MarkovSwitchingRegression(n_states=2, seed=seed).fit(y, factor)
    ms_states = ms_reg.predict_states(y, factor)
    ts(f"  Recovered regime-dependent betas: "
      f"{[round(b[1], 3) for b in ms_reg.beta_]}  (true betas were ~0.4 and ~1.8)")

    # ---- persist reports -------------------------------------------------------
    all_results = {
        "hmm_model_selection": {str(k): v.to_summary_dict() for k, v in selection_results.items()},
        "hmm_best": hmm_result.to_summary_dict(),
        "hmm_validation": validation.to_dict(),
        "gmm": gmm_result.to_summary_dict(),
        "rolling_kmeans": rolling_result.to_summary_dict(),
        "cusum_n_changepoints": len(cps),
        "bocpd_n_changepoints": len(bocpd_out["changepoints"]),
        "markov_switching_betas": [float(b[1]) for b in ms_reg.beta_],
        "ari_hmm_vs_ground_truth": ari_vs_truth,
        "ari_gmm_vs_hmm": ari_gmm_vs_hmm,
    }
    with open(Path(out_dir, "regime_detection_report.json"), "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("REGIME DETECTION SUMMARY")
    print("=" * 78)
    print(f"HMM selected K={best_model.K} states | ARI vs ground truth = {ari_vs_truth:.3f}")
    print(f"HMM average regime durations (days): {hmm_result.average_duration()}")
    print(f"HMM vs GMM label agreement (ARI): {ari_gmm_vs_hmm:.3f}")
    print(f"Rolling K-means silhouette score: {rolling_result.meta['silhouette_score']:.3f}")
    print(f"CUSUM change-points: {len(cps)} | BOCPD change-points: {len(bocpd_out['changepoints'])}")
    print(f"Markov-switching regression recovered betas: "
         f"{[round(b[1], 3) for b in ms_reg.beta_]} (true: ~0.4, ~1.8)")
    print("=" * 78)
    print(f"Plots and JSON report written to: {out_dir}/")
    ts("=== REGIME DETECTION & CLUSTERING DEMO COMPLETE ===")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Regime detection & clustering toolkit")
    p.add_argument("--demo", action="store_true", help="run the synthetic end-to-end demo")
    p.add_argument("--out-dir", type=str, default="./regime_results")
    p.add_argument("--seed", type=int, default=7)
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
