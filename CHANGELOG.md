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
