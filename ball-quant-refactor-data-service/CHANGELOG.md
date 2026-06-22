---
TL;DR: ball-quant gained a complete research harness (capture → settle → backtest → optimize)
on top of the original forward-only pipeline. Runtime stays ZERO third-party deps (load-bearing
invariant). All strategy magic-constants are now in a typed, optimizable `StrategyParams`.
338 tests, 88% coverage. Methodology toggles (Dixon-Coles / Shin devig / inverse-variance) and
per-competition dynamic params added — all default-off and optimizer-tunable. Default model
behavior byte-identical to pre-harness (params defaults == old literals). See docs/HARNESS.md.
---

# CHANGELOG

## 2026-06-14 — Harness framework (v0.2, unreleased)

Upgraded the forward-only betting pipeline into a validatable, optimizable research system.
Built in 6 phases; every phase gated on the prior test suite staying green.

### Added
- **Optimizable strategy core** — `core/params.py::StrategyParams` (frozen dataclass, ~33 fields)
  hoists every magic constant (Poisson prior, calibration schedule, Kelly fraction, correlation
  discounts, budget splits, causal reliability, type-C window…) into one typed, JSON-serializable,
  optimizer-injectable object. `DEFAULT_PARAMS` == the original literals → default behavior unchanged.
- **Data store** — `data/store.py` + `data/capture.py`: lossless snapshot persistence (jsonl +
  manifest) of the raw `EventMarketMatrix` (reuses polymarket's `quote.__dict__` serializer) plus
  captured ticai SP and timestamp, so calibration can be re-run under varied params.
- **Settlement** — `core/settlement.py` + `models.SettlementKey` (now on `Selection`, backfilled in
  `value.selections_from_branches`) + `adapters/results.py`: grade each leg WIN/LOSS/VOID from final
  score, reusing `handicap.handicap_result`. Non-score props grade VOID unless externally resolved.
- **Backtest** — `core/metrics.py` (Brier / log-loss / ECE+reliability / PnL ledger / edge
  realization / Kelly growth, pure `math`) + `backtest/{replay,splits,engine}.py`. `run_backtest`
  replays snapshots → grades → metrics. `splits` enforces no-lookahead (walk-forward / rolling).
- **Optimizer** — `backtest/optimize.py`: grid/random search over `StrategyParams` subspaces, scored
  by walk-forward OUT-OF-SAMPLE metric (selection never by in-sample), deterministic under seed.
  `backtest/report.py` renders backtest + optimization Markdown.
- **Ops** — `config/settings.py` (defaults ← JSON ← `BALLQ_*` env), `logging_setup.py` (additive
  logging at previously-silent adapter error sites; control flow unchanged), 4 CLI subcommands
  `capture / settle / backtest / optimize`, `scripts/`, `docs/HARNESS.md`.
- **Test/CI** — conftest + cassette fixtures, coverage for the 3 previously-zero modules
  (`http` 100%, `api_football` 95%, `markdown` 95%), `.github/workflows/ci.yml`, `.coveragerc`
  (fail_under=75), `Makefile`. Suite: 35 → 274 tests, 88% total coverage.

### Changed
- `pyproject.toml`: added `[project.optional-dependencies] dev`. Removed duplicated/corrupted
  `setup.cfg` (metadata now solely in pyproject). Initialized git (baseline commit `0961e07`).
- Core funcs (`probability/value/combo/staking/causal/analysis`) thread the optional `params` arg.

### Avoided / rejected approaches (don't re-try)
- **Lossy `matrix_to_inventory` as the replay unit** — it drops fields and isn't round-trippable;
  the optimizer would silently calibrate on incomplete data. Use the lossless `write_cached_matrix`
  shape instead (this is why capture stores `matrix` that way).
- **`!= literal` sentinel reconciliation** for params (e.g. `max_goals if max_goals != 7 else …`) —
  fragile and wrong when a caller legitimately passes the default value. Dead override params were
  removed; `StrategyParams` is the single source of truth.
- **Grading Chinese-prose `condition`** — not machine-gradable; added typed `SettlementKey`.

### Known limitations / next steps
- **No real outcomes feed yet.** Backtests need (a) snapshots captured over time and (b) recorded
  final scores. `results.py` reads a CSV (manual); api_football final-score ingestion is optional
  and unbuilt. The repo ships only illustrative demo artifacts under `reports/demo_*.md`.
- **Coverage thin spots:** `causal.py` 59% (reliability-scoring + profile-scale branches),
  `diagnostics.py` 29% (pre-existing network probe), parts of `cli.py`/`polymarket.py` (network,
  omitted from the gate). Not regressions; future test targets.
- `optimize` requires `n_folds + 1` snapshots in range (walk-forward); CLI now emits an actionable
  message rather than a raw traceback.
- `team_total` legs grade VOID unless the entity is normalized to home/away before grading.

## 2026-06-14 (cont.) — dynamic params + methodology toggles (v0.3, unreleased)

### Added
- **Per-competition dynamic params** — `core/profiles.py::ParamProfiles` (global default + per-competition overrides, JSON-persisted) + `resolve(competition) -> StrategyParams`. `run_backtest(..., profiles=...)` resolves params per record's competition; `optimize_by_competition` tunes each competition separately (under-sized groups reported in `skipped`, never silently dropped). CLI: `backtest --profiles`, `optimize --by-competition --profiles-out`. Default (no profiles) → DEFAULT_PARAMS everywhere → behavior unchanged.
- **Methodology toggles in `StrategyParams` (all default-off, optimizer-tunable):**
  - `dixon_coles_rho` (0.0 = off) — Dixon-Coles low-score correction on the Poisson grid.
  - `devig_method` ("proportional" = default | "shin") — Shin (1992) vig removal on the 1X2 devig.
  - `weight_scheme` ("heuristic" = default | "inverse_variance") — spread-based inverse-variance constraint weights.
  These are correct, switchable implementations meant for empirical A/B backtest — NOT claimed to improve calibration. Flip + backtest to decide on your data.

### Fixed
- **Shin toggle was a dead switch end-to-end.** `probability_for_spf` / handicap-fallback called the devig without forwarding `params`, so `devig_method="shin"` never reached `analyze_match` (leaf-function unit tests passed but the wired path silently used the default). Threaded `params` via `ProbabilityContext.params` through all 6 reachable devig call sites + added an end-to-end test (vig'd book → shin ≠ proportional). **Lesson: a toggle needs an end-to-end reachability test, not just a leaf-function test** — caught by a composition smoke, not by the agents' isolated unit tests.

### Known limitations (updated)
- Methodology toggles only move results where they bite: `dixon_coles_rho` / `weight_scheme` shape the **Poisson grid**, which is a **fallback** used only when a market lacks a direct quote — on quote-rich Polymarket data the graded SPF/handicap legs read direct quotes, so those two toggles mainly affect grid-derived markets (totals / correct-score / BTTS) and quote-sparse cases. Shin ≡ proportional on a no-vig book (booksum = 1.0), e.g. the bundled sample data.
- Explicit max-entropy (Lagrange) formulation deferred — IPF already ≈ max-entropy for coherent constraints.

## 2026-06-17 — SKILL.md restructure + session lessons

### Changed
- Restructured `skills/ball-analysis/SKILL.md` from 147 → ~92 lines. Resolved "满买 ↔ 默认不下注" contradiction; updated §2 workflow to `ballq bundle` → LLM flow; added two structural gates to §1.7; deduped R1–R30 rule set; moved session narratives into this CHANGELOG entry.

### Failed approaches / lessons (highest-value, do not repeat)

#### 1. Same-match parlay self-sabotage (anti-correlated legs)
**What happened:** Portugal vs DR Congo. Identified 刚果金+2 as the only +EV leg (P=0.485, edge+3.3%). Attempted to parlay it with 葡萄牙胜 (P=0.765, edge≈−EV) to make the ticket "worth something." The joint condition collapsed to "Portugal wins by exactly 1 goal" (P≈0.25, EV≈−40%).

**Why it broke:** Same-match legs share the same underlying scoreline state space. Adding a second leg from the same match does not diversify — it narrows the winning condition and almost always destroys or inverts the edge sign of the first leg. The two legs are positively correlated in state (both win only on a subset of score outcomes) but the EV interaction is antagonistic.

**Result:** Actual score 1-1. The standalone 刚果金+2 leg won. The parlay (had it been placed) would have lost. The only profit opportunity was the standalone single, which was locked to 仅过关 and thus unplaceable anyway.

**How to avoid:** **within-match = one-bet gate** (now in §1.7 T1). Never parlay two legs from the same match. If same-match multi-玩法 coverage is desired, only single-ticket each玩法 separately (also竞彩 rule: same-match different 玩法 cannot share a 串 leg).

#### 2. 仅过关 lock-out: forcing a −EV 串 when the honest answer is PASS
**What happened:** 刚果金+2 was the sole +EV leg in the card (+3.3%). Its market flag was dg:0, gg:1 — 仅过关, cannot single. Every available parlay partner (葡萄牙胜 @1.27x, etc.) had edge < 0. The temptation was to construct a 串 to "use" the +EV leg.

**Why it broke:** When the only +EV leg is 仅过关-locked, any 串 you build forces at least one −EV partner into the ticket. The combined EV = product of individual EVs (minus 1 normalization), which is always worse than the solo leg. There is no way to recover the +EV by adding −EV legs — they drag the combined EV below zero regardless of the original leg's edge magnitude.

**Result:** Correct action was PASS. The 1-1 result confirmed the standalone 刚果金+2 was right (it won), but there was no legal clean way to bet it. Forcing a 串 would have produced a −EV ticket that also lost (Portugal did not win).

**How to avoid:** **仅过关锁出真实 edge → PASS gate** (now in §1.7 T1). If the only +EV leg is 仅过关 and no other leg clears the edge threshold, the honest action is PASS. Do not construct a −EV 串 just to deploy capital.

#### 3. Moved from SKILL.md: first-round directional prior (narrative)
**06-15 session (all draws):** Four first-round World Cup matches all ended in draws. Fade-热门/受让 tickets won across the board. Tempting to infer "first-round games tend to be tight/draws."

**06-17 session (all blowouts):** Three matches — Austria, Argentina, France — all produced dominant hot-favorite wins (3-1, 3-0, 3-1). Fade-热门 and 受让 tickets failed; 热门胜/穿盘 won.

**双向证伪 conclusion (distilled lesson, kept in SKILL.md §1.6):** Two sessions with completely opposite outcomes = n=1 noise each direction = **no first-round directional prior exists**. The only stable signal across both sessions: **Polymarket胜负排序 was correct both days** (strong team identified correctly). Trust the probability ranking; do not bet a directional "first-round style" prior. Unique structural signal: net-goal incentives push strong teams toward comfortable wins → light bias toward 热门穿盘, never default-fade热门.
