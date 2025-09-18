# owus_ess_competition.py
# based on soil moisture variance (sigma_s) and drydown speed (kappa).

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import math

EPS = 1e-12
ArrayLike = Union[np.ndarray, float, int, None]

@dataclass
class ESSConfig:
    I: float = 0.35         # recharge-like target in [0,1]
    alpha_a: float = 1.0    # maps fww -> a
    alpha_k: float = 1.0    # maps (s* - sw) -> k
    b_const: float = 0.02   # respiration baseline
    mu: float = 0.0         # mutation mixing (index smoother)
    sigma_mut: float = 0.02 # width for mutation smoother (index space)
    beta_sel: float = 1.0   # selection strength multiplier
    max_iter: int = 2000
    tol: float = 1e-10
    # NEW: competition factor hyper-parameters
    eta_var: float = 1.0    # strength of variance penalty (0 = off)
    xi_kappa: float = 0.0   # scaling of penalty by drydown speed (0 = ignore kappa)

def _as_array(x: ArrayLike, N: int) -> np.ndarray:
    if x is None:
        return np.zeros(N)
    if np.isscalar(x):
        return np.full(N, float(x))
    x = np.asarray(x, dtype=float)
    assert x.shape == (N,), "If provided, arrays must have shape (N,)"
    return x

def beta_piecewise(s: float, fww: float, sstar: float, sw: float) -> float:
    if s <= sw:
        return 0.0
    if s <= sstar:
        return fww * (s - sw) / max(sstar - sw, EPS)
    return fww

def resident_env_s_hat(fww: float, sstar: float, sw: float, I: float) -> float:
    if I <= 0.0:
        return 0.0
    if I >= fww - 1e-12:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if beta_piecewise(mid, fww, sstar, sw) < I:
            lo = mid
        else:
            hi = mid
    return max(min(0.5*(lo+hi), 1.0), 0.0)

def owus_to_rgr_knobs(fww: float, sstar: float, sw: float, *, alpha_a: float, alpha_k: float, b_const: float) -> Tuple[float,float,float]:
    a = alpha_a * fww
    k = alpha_k * max(sstar - sw, EPS)
    b = b_const
    return a, k, b

def RGR_at_s(s: float, fww: float, sstar: float, sw: float, *, alpha_a: float, alpha_k: float, b_const: float) -> float:
    a, k, b = owus_to_rgr_knobs(fww, sstar, sw, alpha_a=alpha_a, alpha_k=alpha_k, b_const=b_const)
    x = max(s - sw, 0.0)
    return a * (x / (k + x + EPS)) - b

# Standard normal CDF
def Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def competition_factor(inv_sw: float, env_mu: float, env_sigma: float, kappa: float,
                       eta_var: float, xi_kappa: float) -> float:
    """
    CF = max(0, 1 - eta_var * H)
    where H = Phi((sw_inv - mu_env)/sigma_env) * (1 + xi_kappa * kappa_env).
    Intuition: if invader’s sw is high relative to environment mean (in std units),
    chance of shutdown increases -> stronger penalty; faster drydown (kappa) amplifies it.
    """
    if env_sigma <= 0.0:
        H = 1.0 if inv_sw > env_mu else 0.0
    else:
        z = (inv_sw - env_mu) / env_sigma
        H = Phi(z)
    H *= (1.0 + xi_kappa * max(kappa, 0.0))
    CF = max(0.0, 1.0 - eta_var * H)
    return CF

def build_payoff_matrix(fww: np.ndarray, sstar: np.ndarray, sw: np.ndarray, cfg: ESSConfig,
                        sigma_s: ArrayLike = None, kappa: ArrayLike = None) -> Tuple[np.ndarray, np.ndarray]:
    N = len(fww)
    s_hat = np.array([resident_env_s_hat(fww[i], sstar[i], sw[i], cfg.I) for i in range(N)])
    sig = _as_array(sigma_s, N)
    kap = _as_array(kappa, N)

    E = np.zeros((N, N), dtype=float)
    for i in range(N):
        mu_i, sig_i, kap_i = s_hat[i], sig[i], kap[i]
        for j in range(N):
            base = RGR_at_s(mu_i, fww[j], sstar[j], sw[j],
                            alpha_a=cfg.alpha_a, alpha_k=cfg.alpha_k, b_const=cfg.b_const)
            CF = competition_factor(sw[j], mu_i, sig_i, kap_i, cfg.eta_var, cfg.xi_kappa)
            E[j, i] = cfg.beta_sel * base * CF
    return E, s_hat

def replicator(E: np.ndarray, mu: float = 0.0, max_iter: int = 2000, tol: float = 1e-10) -> np.ndarray:
    N = E.shape[0]
    p = np.ones(N) / N
    if mu > 0:
        K = np.zeros((N, N))
        for i in range(N):
            K[i, i] = 1.0
            if i > 0:   K[i-1, i] += 0.5
            if i < N-1: K[i+1, i] += 0.5
        K /= K.sum(axis=0, keepdims=True) + EPS
    else:
        K = None

    for _ in range(max_iter):
        fitness = E @ p
        Fbar = float(np.dot(p, fitness))
        if Fbar <= 0:
            break
        p_next = p * fitness / Fbar
        if mu > 0:
            p_next = (1 - mu) * p_next + mu * (K @ p_next)
        p_next = np.clip(p_next, 0, None)
        S = p_next.sum()
        if S == 0:
            break
        p_next /= S
        if np.max(np.abs(p_next - p)) < tol:
            return p_next
        p = p_next
    return p

def ess_indices(E: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    N = E.shape[0]
    diag = np.diag(E)
    ok = np.ones(N, dtype=bool)
    for i in range(N):
        if np.any(diag[i] < E[:, i] - tol):
            ok[i] = False
    return np.where(ok)[0]

def summarize_distribution_sw(sw: np.ndarray, weights: np.ndarray,
                              grid: Optional[np.ndarray] = None, bw: float = 0.02) -> Dict[str, np.ndarray]:
    if grid is None:
        grid = np.linspace(max(sw.min()-0.05, 0.0), min(sw.max()+0.05, 1.0), 200)
    diffs = grid[:, None] - sw[None, :]
    W = np.exp(-0.5 * (diffs / (bw + EPS))**2)
    dens = (W * weights[None, :]).sum(axis=1)
    area = np.trapz(dens, grid) + EPS
    dens /= area
    mean = float(np.trapz(grid * dens, grid))
    var  = float(np.trapz((grid - mean)**2 * dens, grid))
    return {"grid": grid, "density": dens, "mean": mean, "var": var}

def run_owus_ess(fww: np.ndarray, sstar: np.ndarray, sw: np.ndarray, cfg: Optional[ESSConfig] = None,
                 sigma_s: ArrayLike = None, kappa: ArrayLike = None, make_plots: bool = True) -> Dict[str, np.ndarray]:
    if cfg is None:
        cfg = ESSConfig()
    assert fww.shape == sstar.shape == sw.shape, "All inputs must have same shape."
    N = len(fww)

    E, s_hat = build_payoff_matrix(fww, sstar, sw, cfg, sigma_s=sigma_s, kappa=kappa)
    ess_idx = ess_indices(E)
    p = replicator(E, mu=cfg.mu, max_iter=cfg.max_iter, tol=cfg.tol)

    prior_sw = np.ones(N)/N
    prior_kde = summarize_distribution_sw(sw, prior_sw)
    post_kde  = summarize_distribution_sw(sw, p, grid=prior_kde["grid"])

    if make_plots:
        plt.figure(figsize=(5,4))
        plt.imshow(E, origin="lower", cmap="viridis")
        plt.colorbar(label="Payoff E(invader, resident) with competition factor")
        plt.title("Payoff matrix with variance/drydown competition")
        plt.xlabel("resident index"); plt.ylabel("invader index")
        plt.tight_layout()

        plt.figure(figsize=(7,4))
        plt.plot(prior_kde["grid"], prior_kde["density"], label="Innate $p_0(s_w)$")
        plt.plot(post_kde["grid"],  post_kde["density"],  label="Replicator mixture $p(s_w)$")
        if len(ess_idx) > 0:
            for i in ess_idx:
                plt.axvline(sw[i], color="k", alpha=0.25, linestyle="--")
        plt.xlabel("$s_w$"); plt.ylabel("density")
        plt.title("Innate vs. emergent mixture over $s_w$ (with competition factor)")
        plt.legend(); plt.tight_layout()
        plt.show()

    return {"E": E, "s_hat": s_hat, "p": p, "ess_idx": ess_idx}

if __name__ == "__main__":
    # Small synthetic demo
    rng = np.random.d
