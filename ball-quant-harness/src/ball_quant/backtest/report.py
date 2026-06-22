"""
Markdown report renderers for backtest and optimization results.

Both functions accept plain dicts and render gracefully when blocks are
empty — no crash on missing calibration/pnl/kelly data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ball_quant.core.params import DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any, fmt_spec: str = ".4f") -> str:
    """Format a numeric value or return '—' if it is None / non-numeric."""
    if value is None:
        return "—"
    try:
        return format(float(value), fmt_spec)
    except (TypeError, ValueError):
        return "—"


def _pct(value: Any) -> str:
    """Format as percentage with 2 decimal places, or '—'."""
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Backtest report
# ---------------------------------------------------------------------------

def render_backtest_report(result: dict, title: str = "Backtest Report") -> str:
    """Render a Markdown backtest report from a run_backtest result dict.

    Gracefully renders '—' for any empty blocks — will not crash if pnl,
    calibration, edge, or kelly blocks are absent.
    """
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")

    # --- summary ---
    lines.append("## 总览 (Summary)")
    lines.append("")
    lines.append(f"- 记录数 (records): {result.get('n_records', '—')}")
    lines.append(f"- 已定级比赛 (graded matches): {result.get('n_graded_matches', '—')}")
    lines.append(f"- 跳过 (no outcome): {result.get('skipped_no_outcome', '—')}")
    lines.append(f"- 跳过 (no SP): {result.get('skipped_no_sp', '—')}")
    lines.append(f"- 校准点数 (calibration points): {result.get('n_calibration_points', '—')}")
    lines.append(f"- 注单数 (bets): {result.get('n_bets', '—')}")
    lines.append("")

    metrics = result.get("metrics", {})

    # --- calibration table ---
    lines.append("## 校准指标 (Calibration)")
    lines.append("")
    calib = metrics.get("calibration", {})
    n_pts = result.get("n_calibration_points", "—")
    lines.append("| 指标 | 值 | N |")
    lines.append("|---|---:|---:|")
    lines.append(f"| brier | {_fmt(calib.get('brier'))} | {n_pts} |")
    lines.append(f"| log_loss | {_fmt(calib.get('log_loss'))} | {n_pts} |")
    lines.append(f"| ece | {_fmt(calib.get('ece'))} | {n_pts} |")
    lines.append("")

    # --- pnl table ---
    lines.append("## 损益台账 (PnL Ledger)")
    lines.append("")
    pnl = metrics.get("pnl", {})
    if pnl:
        lines.append("| 指标 | 值 |")
        lines.append("|---|---:|")
        lines.append(f"| net_pnl | {_fmt(pnl.get('net_pnl'))} |")
        lines.append(f"| roi | {_pct(pnl.get('roi'))} |")
        lines.append(f"| n_bets | {pnl.get('n_bets', '—')} |")
        lines.append(f"| n_win | {pnl.get('n_win', '—')} |")
        lines.append(f"| n_loss | {pnl.get('n_loss', '—')} |")
        lines.append(f"| n_void | {pnl.get('n_void', '—')} |")
        lines.append(f"| total_stake_at_risk | {_fmt(pnl.get('total_stake_at_risk'))} |")
        lines.append(f"| total_return | {_fmt(pnl.get('total_return'))} |")
    else:
        lines.append("_无注单数据 (no bets graded)_")
    lines.append("")

    # --- edge block ---
    lines.append("## 边际分析 (Edge)")
    lines.append("")
    edge = metrics.get("edge", {})
    if edge:
        lines.append(f"- mean predicted edge: {_pct(edge.get('mean_predicted_edge'))}")
        lines.append(f"- mean realized return/unit: {_pct(edge.get('mean_realized_return_per_unit'))}")
        lines.append(f"- correlation (pred vs realized): {_fmt(edge.get('correlation'))}")
        lines.append(f"- n: {edge.get('n', '—')}")
    else:
        lines.append("_无注单数据 (no bets)_")
    lines.append("")

    # --- kelly block ---
    lines.append("## Kelly 成长率 (Kelly Growth)")
    lines.append("")
    kelly = metrics.get("kelly", {})
    if kelly:
        lines.append(f"- geometric growth rate: {_pct(kelly.get('geometric_growth_rate'))}")
        lines.append(f"- sum log growth: {_fmt(kelly.get('sum_log_growth'))}")
        lines.append(f"- n: {kelly.get('n', '—')}")
    else:
        lines.append("_无注单数据 (no bets)_")
    lines.append("")

    # --- per-market-type brier table ---
    per_mt = result.get("per_market_type", {})
    if per_mt:
        lines.append("## 各市场类型 Brier 分布 (Per Market Type)")
        lines.append("")
        lines.append("| 市场类型 | brier | n |")
        lines.append("|---|---:|---:|")
        for mt, info in sorted(per_mt.items()):
            lines.append(f"| {mt} | {_fmt(info.get('brier'))} | {info.get('n', '—')} |")
        lines.append("")

    # --- reliability bins ---
    reliability = calib.get("reliability", [])
    if reliability:
        lines.append("## 可靠性分布 (Reliability Bins)")
        lines.append("")
        lines.append("| 区间 | 均预测概率 | 实际频率 | n |")
        lines.append("|---|---:|---:|---:|")
        for b in reliability:
            bin_label = f"[{b['bin_lo']:.2f}, {b['bin_hi']:.2f})"
            lines.append(
                f"| {bin_label} | {_fmt(b.get('mean_pred'))} | {_fmt(b.get('emp_freq'))} | {b.get('n', '—')} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimization report
# ---------------------------------------------------------------------------

def render_optimization_report(opt: dict, title: str = "Optimization Report") -> str:
    """Render a Markdown optimization report from an optimize_params result dict.

    Shows: metric/direction/search/n_trials; BEST overrides table with default
    vs optimized values; best IS vs OOS + overfit gap warning; top-10 trials.
    """
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")

    # --- header block ---
    lines.append("## 搜索配置 (Search Config)")
    lines.append("")
    lines.append(f"- metric: **{opt.get('metric', '—')}** ({opt.get('direction', '—')})")
    lines.append(f"- search: {opt.get('search', '—')}")
    lines.append(f"- n_trials: {opt.get('n_trials', '—')}")
    lines.append(f"- n_folds: {opt.get('n_folds', '—')}")
    lines.append("")

    # --- best params table ---
    lines.append("## 最优参数 (Best Overrides)")
    lines.append("")
    best_overrides = opt.get("best_overrides", {})
    default_dict = DEFAULT_PARAMS.to_dict()

    lines.append("| 字段 (field) | 默认值 (default) | 最优值 (optimized) |")
    lines.append("|---|---:|---:|")
    for field, optimized_val in sorted(best_overrides.items()):
        default_val = default_dict.get(field, "—")
        lines.append(f"| {field} | {default_val} | {optimized_val} |")
    lines.append("")

    # --- IS vs OOS comparison ---
    lines.append("## 样本内外对比 (In-Sample vs Out-of-Sample)")
    lines.append("")
    best_is = opt.get("best_in_sample")
    best_oos = opt.get("best_out_of_sample")
    overfit_gap = opt.get("overfit_gap")

    lines.append(f"- best in-sample: {_fmt(best_is)}")
    lines.append(f"- best out-of-sample: {_fmt(best_oos)}")
    lines.append(f"- overfit gap (IS - OOS for max; OOS - IS for min): {_fmt(overfit_gap)}")

    # Warn when OOS is markedly worse than IS.
    # "Markedly worse" = |gap| > 0.05 for min metrics (brier/log_loss/ece scale 0-1)
    # or > 5% relative difference for max metrics (pnl/roi can be unbounded).
    if overfit_gap is not None and overfit_gap > 0.05:
        direction = opt.get("direction", "")
        if direction == "min":
            lines.append("")
            lines.append(
                "> ⚠ overfit_gap > 0.05: out-of-sample calibration is markedly worse "
                "than in-sample. Consider fewer trials or a wider regularisation prior."
            )
        else:
            lines.append("")
            lines.append(
                "> ⚠ overfit_gap > 0.05: out-of-sample return is markedly lower than "
                "in-sample. Walk-forward leakage check recommended."
            )
    lines.append("")

    # --- top-10 trials ---
    lines.append("## 前10 试验 (Top-10 Trials by OOS)")
    lines.append("")

    trials = opt.get("trials", [])
    direction = opt.get("direction", "max")

    # Sort by OOS: None goes to the bottom regardless of direction.
    def _sort_key(t: dict):
        oos = t.get("out_of_sample")
        if oos is None:
            # Put None at the very end: use worst sentinel.
            return (1, 0.0)
        if direction == "min":
            return (0, oos)   # ascending: smaller is better
        return (0, -oos)      # descending: larger is better

    sorted_trials = sorted(trials, key=_sort_key)[:10]

    if sorted_trials:
        # Build a compact header from the override keys of the first trial.
        override_keys = sorted(sorted_trials[0].get("overrides", {}).keys())
        col_headers = " | ".join(override_keys)
        sep_cols = " | ".join(["---:" for _ in override_keys])
        lines.append(f"| # | {col_headers} | in_sample | out_of_sample | undefined |")
        lines.append(f"|---|{sep_cols}|---:|---:|---|")
        for rank, trial in enumerate(sorted_trials, 1):
            ovr = trial.get("overrides", {})
            vals = " | ".join(str(ovr.get(k, "—")) for k in override_keys)
            is_val = _fmt(trial.get("in_sample"))
            oos_val = _fmt(trial.get("out_of_sample"))
            undef = "yes" if trial.get("undefined") else "no"
            lines.append(f"| {rank} | {vals} | {is_val} | {oos_val} | {undef} |")
    else:
        lines.append("_无试验数据 (no trials)_")
    lines.append("")

    return "\n".join(lines)
