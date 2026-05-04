#!/usr/bin/env python3
"""Parametric pseudo-catalog forecast for posterior-predictive A.

Model:
    v_Lambda = sigma_Lambda^2
    A | mu_Lambda, v_Lambda ~ Normal(mu_Lambda, v_Lambda)

For each total catalog size N:
    s_mu(N) = s_mu_ref * (N_ref / N)**mu_scale_power
    s_v(N)  = s_v_ref  * (N_ref / N)**sigma2_scale_power

A pseudo-catalog draw is:
    mu_hat ~ Normal(mu_truth, s_mu(N)^2)
    v_hat  ~ Normal_+(v_truth, s_v(N)^2)
    mu_Lambda_s ~ Normal(mu_hat, s_mu(N)^2)
    v_Lambda_s  ~ q_+(v_hat, s_v(N))
    A_s ~ Normal(mu_Lambda_s, v_Lambda_s)

The default uses truncated normals for both v_hat and v_Lambda because the GR
truth v=0 is a boundary; an inverse-chi-square posterior approximation for
v_Lambda is available with --sigma2-posterior invchi2.

Outputs:
    forecast_A_parametric.csv
    forecast_A_parametric.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TARGET = 1.0
DEFAULT_NSAMP = 20_000
DEFAULT_N_REPS = 100
DEFAULT_N_POINTS = 80
DEFAULT_MAX_TOTAL_N = 100_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--posterior-nc", required=True, help="Hierarchical posterior NetCDF file.")
    p.add_argument("--outdir", required=True, help="Directory for forecast CSV/PNG.")

    nsrc = p.add_mutually_exclusive_group(required=True)
    nsrc.add_argument("--n-ref", type=int, help="Reference/current number of analyzed events.")
    nsrc.add_argument("--include-events-file", help="Text file with one analyzed event name per line.")

    p.add_argument("--mu-var", default="mu_tgr", help="Name of mu_Lambda variable in posterior NetCDF.")
    p.add_argument("--sigma-var", default="sigma_tgr", help="Name of sigma_Lambda variable in posterior NetCDF.")
    p.add_argument("--mu-truth", type=float, default=1.0, help="Forecast truth for mu_Lambda. Default: GR value 1.")
    p.add_argument("--sigma2-truth", type=float, default=0.0, help="Forecast truth for v_Lambda=sigma_Lambda^2. Default: GR value 0.")
    p.add_argument("--target", type=float, default=DEFAULT_TARGET, help="Target uncertainty on A. Default: 1.")
    p.add_argument("--metric", choices=["halfwidth68", "std"], default="halfwidth68")
    p.add_argument("--scale-estimator", choices=["std", "halfwidth68"], default="halfwidth68",
                   help="How to estimate current posterior widths of mu_Lambda and v_Lambda.")
    p.add_argument("--mu-scale-power", type=float, default=0.5,
                   help="p_mu in s_mu(N)=s_mu_ref*(N_ref/N)^p_mu. Default 0.5 gives 1/sqrt(N) width scaling.")
    p.add_argument("--sigma2-scale-power", type=float, default=0.5,
                   help="p_v in s_v(N)=s_v_ref*(N_ref/N)^p_v. Default 0.5 gives 1/sqrt(N) width scaling for v=sigma^2. Use 1.0 to make the sqrt(v) contribution to A shrink roughly as 1/sqrt(N) near v_truth=0.")
    p.add_argument("--sigma2-posterior", choices=["truncnorm", "invchi2"], default="truncnorm",
                   help="Positive model for posterior draws of v_Lambda around v_hat. Default: truncnorm. invchi2 is optional but can be heavy-tailed near v_hat~0.")
    p.add_argument("--nsamp", type=int, default=DEFAULT_NSAMP, help="Posterior-predictive samples per pseudo-catalog.")
    p.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS, help="Pseudo-catalog repetitions per N.")
    p.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS, help="Number of N values in output grid.")
    p.add_argument("--n-max", type=int, default=None, help="Maximum total N in output grid. Auto-chosen if omitted.")
    p.add_argument("--max-total-n", type=int, default=DEFAULT_MAX_TOTAL_N, help="Hard cap for automatic n-max.")
    p.add_argument("--max-posterior-draws", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def count_events(path: str | Path) -> int:
    events: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                events.add(line)
    if not events:
        raise RuntimeError(f"No events found in {path}.")
    return len(events)


def load_hyperposterior(nc_path: str | Path, mu_var: str, sigma_var: str, max_draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fit = az.from_netcdf(nc_path)
    posterior = fit.posterior
    if mu_var not in posterior:
        raise KeyError(f"Could not find {mu_var!r} in posterior NetCDF.")
    if sigma_var not in posterior:
        raise KeyError(f"Could not find {sigma_var!r} in posterior NetCDF.")

    mu = np.asarray(posterior[mu_var].values, dtype=float).reshape(-1)
    sigma = np.asarray(posterior[sigma_var].values, dtype=float).reshape(-1)
    good = np.isfinite(mu) & np.isfinite(sigma) & (sigma >= 0.0)
    mu, sigma = mu[good], sigma[good]
    if len(mu) < 2:
        raise RuntimeError("Need at least two finite posterior samples.")

    if max_draws is not None and len(mu) > max_draws:
        idx = rng.choice(len(mu), size=max_draws, replace=False)
        mu, sigma = mu[idx], sigma[idx]
    return mu, sigma**2


def width(x: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if method == "std":
        return float(np.std(x, ddof=1))
    q16, q84 = np.quantile(x, [0.16, 0.84])
    return 0.5 * float(q84 - q16)


def q1684(x: np.ndarray) -> tuple[float, float, float]:
    q16, q50, q84 = np.quantile(np.asarray(x, dtype=float), [0.16, 0.50, 0.84])
    return float(q16), float(q50), float(q84)


def rtruncnorm_pos(rng: np.random.Generator, loc: float, scale: float, size: int) -> np.ndarray:
    """Draw Normal(loc, scale^2) truncated to x>=0. For loc=0 this is half-normal."""
    loc, scale = float(loc), float(scale)
    if scale <= 0.0:
        return np.full(size, max(loc, 0.0))
    if loc == 0.0:
        return np.abs(rng.normal(0.0, scale, size=size))

    out = rng.normal(loc, scale, size=size)
    bad = out < 0.0
    tries = 0
    while np.any(bad):
        out[bad] = rng.normal(loc, scale, size=int(np.sum(bad)))
        bad = out < 0.0
        tries += 1
        if tries > 1000:
            raise RuntimeError("truncated-normal rejection sampling failed")
    return out


def rinvchi2_mean_sd(rng: np.random.Generator, mean: float, sd: float, size: int) -> np.ndarray:
    """Moment-matched inverse-gamma/scaled-inverse-chi-square draw for v>0.

    This is useful only when mean is appreciably positive. Near mean=0 with
    nonzero sd it becomes extremely heavy-tailed, which is why it is not the
    default for this forecast.
    """
    mean, sd = float(mean), float(sd)
    if sd <= 0.0:
        return np.full(size, max(mean, 0.0))
    if mean <= 0.0:
        return np.zeros(size)
    alpha = 2.0 + mean * mean / (sd * sd)
    beta = mean * (alpha - 1.0)
    y = rng.gamma(shape=alpha, scale=1.0 / beta, size=size)  # Gamma(rate=beta)
    return 1.0 / np.maximum(y, np.finfo(float).tiny)


def draw_v_posterior(rng: np.random.Generator, v_hat: float, s_v: float, nsamp: int, model: str) -> np.ndarray:
    if model == "invchi2":
        return rinvchi2_mean_sd(rng, v_hat, s_v, nsamp)
    return rtruncnorm_pos(rng, v_hat, s_v, nsamp)


def simulate_one(rng: np.random.Generator, s_mu: float, s_v: float, mu_truth: float, v_truth: float, nsamp: int, metric: str, v_model: str) -> dict[str, float]:
    mu_hat = float(rng.normal(mu_truth, s_mu))
    # v_hat is a noisy nonnegative estimator centered on a boundary. It is not
    # modeled as inverse-chi-square because inverse-chi-square cannot be centered
    # at v_truth=0 except as a degenerate distribution.
    v_hat = float(rtruncnorm_pos(rng, v_truth, s_v, 1)[0])

    mu_lam = rng.normal(mu_hat, s_mu, size=nsamp)
    v_lam = draw_v_posterior(rng, v_hat, s_v, nsamp, v_model)
    a = mu_lam + np.sqrt(np.maximum(v_lam, 0.0)) * rng.normal(size=nsamp)

    a16, a50, a84 = q1684(a)
    return {
        "mu_hat": mu_hat,
        "sigma2_hat": v_hat,
        "A_q16": a16,
        "A_q50": a50,
        "A_q84": a84,
        "A_halfwidth68": 0.5 * (a84 - a16),
        "A_std": float(np.std(a, ddof=1)),
        "metric_value": float(np.std(a, ddof=1)) if metric == "std" else 0.5 * (a84 - a16),
        "mean_sigma2_posterior": float(np.mean(v_lam)),
    }


def summarize_at_n(rng: np.random.Generator, n_ref: int, n_total: int, s_mu_ref: float, s_v_ref: float, mu_power: float, v_power: float, mu_truth: float, v_truth: float, nsamp: int, n_reps: int, metric: str, v_model: str, target: float) -> dict[str, float | int]:
    f_mu = (float(n_ref) / float(n_total)) ** float(mu_power)
    f_v = (float(n_ref) / float(n_total)) ** float(v_power)
    s_mu, s_v = s_mu_ref * f_mu, s_v_ref * f_v

    reps = pd.DataFrame([
        simulate_one(rng, s_mu, s_v, mu_truth, v_truth, nsamp, metric, v_model)
        for _ in range(n_reps)
    ])

    return {
        "n_total": int(n_total),
        "mu_shrink_factor": float(f_mu),
        "sigma2_shrink_factor": float(f_v),
        "s_mu": float(s_mu),
        "s_sigma2": float(s_v),
        "metric_p16": float(reps["metric_value"].quantile(0.16)),
        "metric_p50": float(reps["metric_value"].quantile(0.50)),
        "metric_p84": float(reps["metric_value"].quantile(0.84)),
        "meets_target_fraction": float(np.mean(reps["metric_value"] <= target)),
        "A_halfwidth68_p50": float(reps["A_halfwidth68"].quantile(0.50)),
        "A_std_p50": float(reps["A_std"].quantile(0.50)),
        "A_q16_p50": float(reps["A_q16"].quantile(0.50)),
        "A_q50_p50": float(reps["A_q50"].quantile(0.50)),
        "A_q84_p50": float(reps["A_q84"].quantile(0.50)),
        "mu_hat_p50": float(reps["mu_hat"].quantile(0.50)),
        "sigma2_hat_p50": float(reps["sigma2_hat"].quantile(0.50)),
        "mean_sigma2_posterior_p50": float(reps["mean_sigma2_posterior"].quantile(0.50)),
    }


def current_predictive_width(rng: np.random.Generator, mu: np.ndarray, v: np.ndarray, nsamp: int, metric: str) -> float:
    idx = rng.choice(len(mu), size=nsamp, replace=(len(mu) < nsamp))
    a = mu[idx] + np.sqrt(np.maximum(v[idx], 0.0)) * rng.normal(size=nsamp)
    return width(a, "std" if metric == "std" else "halfwidth68")


def choose_grid(n_ref: int, target: float, current_width: float, mu_power: float, v_power: float, n_points: int, n_max: int | None, max_total_n: int) -> np.ndarray:
    if n_max is None:
        # The A width may shrink like N^{-min(mu_power, v_power/2)} near v_truth=0.
        eff_power = max(min(float(mu_power), float(v_power) / 2.0), 1e-6)
        guess = n_ref * (current_width / target) ** (1.0 / eff_power) if current_width > target else 10 * n_ref
        n_max = int(min(max_total_n, max(10 * n_ref, 2.0 * guess)))
    n_max = max(int(n_max), int(n_ref))
    grid = np.unique(np.round(np.geomspace(n_ref, n_max, max(n_points, 2))).astype(int))
    return np.unique(np.concatenate(([n_ref], grid))).astype(int)


def make_plot(df: pd.DataFrame, n_ref: int, target: float, metric: str, n_req: int | None, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = df["n_total"].to_numpy(float)
    ax.fill_between(x, df["metric_p16"], df["metric_p84"], alpha=0.15, label="pseudo-catalog p16-p84")
    ax.plot(x, df["metric_p50"], marker="o", markersize=3, label="pseudo-catalog median")
    ax.axhline(target, linestyle="--", linewidth=1.2, label=f"target = {target:g}")
    ax.axvline(n_ref, linestyle=":", linewidth=1.2, label=f"reference N = {n_ref}")
    if n_req is not None:
        ax.axvline(n_req, linestyle="-.", linewidth=1.2, label=f"median crossing N = {n_req}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total number of analyzed detections")
    ax.set_ylabel(f"A uncertainty metric: {metric}")
    ax.set_title("Parametric posterior-predictive forecast for A")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.nsamp <= 1 or args.n_reps <= 0:
        raise ValueError("--nsamp must be >1 and --n-reps must be positive.")
    if args.target <= 0 or args.sigma2_truth < 0:
        raise ValueError("--target must be positive and --sigma2-truth nonnegative.")

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_ref = int(args.n_ref) if args.n_ref is not None else count_events(args.include_events_file)
    mu, v = load_hyperposterior(args.posterior_nc, args.mu_var, args.sigma_var, args.max_posterior_draws, rng)

    s_mu_ref = width(mu, args.scale_estimator)
    s_v_ref = width(v, args.scale_estimator)
    current_A_width = current_predictive_width(rng, mu, v, args.nsamp, args.metric)
    pure_sqrtn_from_current_A = n_ref * (current_A_width / args.target) ** 2
    pure_sqrtn_from_param_mu = n_ref * (s_mu_ref / args.target) ** 2

    n_grid = choose_grid(n_ref, args.target, current_A_width, args.mu_scale_power, args.sigma2_scale_power, args.n_points, args.n_max, args.max_total_n)
    rows = [
        summarize_at_n(
            rng, n_ref, int(n), s_mu_ref, s_v_ref, args.mu_scale_power, args.sigma2_scale_power,
            args.mu_truth, args.sigma2_truth, args.nsamp, args.n_reps, args.metric, args.sigma2_posterior, args.target
        )
        for n in n_grid
    ]
    df = pd.DataFrame(rows)
    crossings = df.loc[df["metric_p50"] <= args.target, "n_total"]
    n_req = None if crossings.empty else int(crossings.iloc[0])

    df["median_meets_target"] = df["metric_p50"] <= args.target
    df["target"] = float(args.target)
    df["metric"] = args.metric
    df["n_ref"] = int(n_ref)
    df["mu_truth"] = float(args.mu_truth)
    df["sigma2_truth"] = float(args.sigma2_truth)
    df["s_mu_ref"] = float(s_mu_ref)
    df["s_sigma2_ref"] = float(s_v_ref)
    df["scale_estimator"] = args.scale_estimator
    df["mu_scale_power"] = float(args.mu_scale_power)
    df["sigma2_scale_power"] = float(args.sigma2_scale_power)
    df["sigma2_posterior"] = args.sigma2_posterior
    df["current_empirical_A_width"] = float(current_A_width)
    df["pure_sqrtn_N_from_current_A"] = float(pure_sqrtn_from_current_A)
    df["pure_sqrtn_N_from_mu_width"] = float(pure_sqrtn_from_param_mu)
    df["n_required_median_grid"] = np.nan if n_req is None else n_req
    df["posterior_draws_used"] = int(len(mu))
    df["nsamp"] = int(args.nsamp)
    df["n_reps"] = int(args.n_reps)

    csv_path = outdir / "forecast_A_parametric.csv"
    png_path = outdir / "forecast_A_parametric.png"
    df.to_csv(csv_path, index=False)
    make_plot(df, n_ref, args.target, args.metric, n_req, png_path)

    print(f"Reference/current N: {n_ref}")
    print(f"Reference scales ({args.scale_estimator}): s_mu={s_mu_ref:.6g}, s_sigma2={s_v_ref:.6g}")
    print(f"Current empirical posterior-predictive A {args.metric}: {current_A_width:.6g}")
    print(f"Pure 1/sqrt(N) crossing from current A width: {pure_sqrtn_from_current_A:.1f}")
    print(f"Pure 1/sqrt(N) crossing from mu width only: {pure_sqrtn_from_param_mu:.1f}")
    print(f"Scale laws: s_mu~(N_ref/N)^{args.mu_scale_power:g}, s_sigma2~(N_ref/N)^{args.sigma2_scale_power:g}")
    print(f"sigma2 posterior model: {args.sigma2_posterior}")
    if n_req is None:
        print(f"No median crossing of target={args.target:g} found on grid up to N={int(n_grid[-1])}.")
    else:
        print(f"First grid N with median A {args.metric} <= {args.target:g}: {n_req}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
