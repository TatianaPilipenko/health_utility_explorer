# QALY/NHANES Script — Concrete Refactor Plan

**Context:** The script runs in **Google Colab** with **@param** form fields as the interactive interface. The data file path is Colab/Drive; the script was copied to local for version control. This plan keeps Colab + @param as the runtime and UI, and focuses on structure, correctness, and maintainability.

---

## Principles

1. **Colab-first:** Entry point remains a single `qaly()` (or thin wrapper) called at the bottom; @param widgets stay as the user interface.
2. **No behaviour change by default:** Refactors should preserve current scoring logic until explicitly replaced (e.g. config-driven weights).
3. **Incremental:** Work in phases so the notebook stays runnable after each phase.
4. **Testable:** Extract pure functions so they can be unit-tested with small inputs.

---

## Phase 1: Data loading and cleaning (single module)

**Goal:** One place that loads the CSV, normalizes missing values, enforces types, and returns a clean dataframe. Rest of the code uses this instead of ad-hoc cleaning.

### 1.1 Current problems

| Issue | Location / symptom | Correction |
|-------|--------------------|------------|
| **Path hard-coded** | `file_path = '/content/drive/...'` | Keep for Colab; add optional override (e.g. from env or a single config at top) so the same script can run locally with a different path. |
| **Missing-value strings not normalized** | Only `["no info", "-"]` replaced; CSV may have "No info", "NA", etc. | Use a single `na_values` list (and optionally NHANES special codes 7, 9, 7777, 9999) in one place. |
| **Mixed dtypes** | `DtypeWarning` for columns 9,10,12,17; many numeric cols as `object`. | Explicit `dtype` or `pd.to_numeric(..., errors='coerce')` for all intended numeric columns on read or in a dedicated cleaning step. |
| **SEQN as float** | Respondent ID should be integer. | Cast `SEQN` to `Int64` (nullable) or drop nulls and use `int`. |
| **Incomplete numeric list** | Only `numeric_columns` (lines 245–250) are coerced; `PAQ655`, `PAQ670`, `PAD680` are in `benchmark_metrics` but not in `numeric_columns`. | Either add them to `numeric_columns` or derive “numeric columns” from a single schema (see Phase 2). |
| **Categorical columns** | `categorical_mappings` use int keys (1, 2, 7, 9); after `to_numeric(..., errors='coerce')`, 7/9 become NaN. Replace with labels on a **copy** (`nhanes_data_copy`), but **raw** `nhanes_data` is still used in the big `apply` (lines 2703–2752). | Use one canonical representation: either (A) keep numeric codes everywhere and compare to 1/2 in the apply, or (B) apply the same mapping to the dataframe used for contribution computation so `row['BPQ020'] == 'Yes'` is valid. Currently (B) is intended but the apply runs on unmapped data → **bug**. |

### 1.2 Improvements

- **Single loader function**  
  `load_and_clean_nhanes(file_path: str, columns_to_drop: list) -> pd.DataFrame`  
  - Reads CSV with `na_values`, optional `low_memory=False` to reduce DtypeWarning.  
  - Drops `columns_to_drop`.  
  - Applies numeric coercion for all columns that should be numeric (from a shared list or schema).  
  - Casts `SEQN` appropriately.  
  - Returns one dataframe; no separate “copy” for categoricals unless needed for a specific reason (then document why).

- **Single place for “raw → label” for categoricals**  
  - If downstream logic uses string labels ("Yes"/"No"), apply the same `categorical_mappings` to the **same** dataframe that is used for the row-wise contribution `apply`, so that comparisons like `row['BPQ020'] == 'Yes'` are correct.  
  - Alternatively, keep numeric codes everywhere and compare `row['BPQ020'] == 1` (and treat 7, 9 as missing). Decide one convention and use it consistently.

- **Optional local path**  
  At the top of the script (or in a small config block):  
  `DATA_PATH = os.environ.get("QALY_NHANES_CSV", "/content/drive/MyDrive/...")`  
  So in Colab nothing changes; locally you can set the env var.

---

## Phase 2: Configuration (weights, boundaries, mappings)

**Goal:** All magic numbers, healthy/unhealthy ranges, and categorical mappings live in one or two structured objects (e.g. dicts or a small config file). No scattered literals inside the scoring logic.

### 2.1 Current problems

| Issue | Location / symptom | Correction |
|-------|--------------------|------------|
| **Weights and thresholds scattered** | `numeric_boundaries`, `categorical_adjustments`, `baseline_hdl`, `baseline_tc`, sigmoid params (`k_low`, `k_high`), `scaling_factor = 0.05`, `MIN_UTILITY_SCORE`, `MAX_UTILITY_SCORE`, `benchmark_age = 80`, and per-contribution multipliers (1.2, 1.3, 1.4, 1.5) are spread across the file. | Move into a single config structure (see below). |
| **Bug: missing key in baseline_tc** | When `nhanes_mean is None`, code uses `thresholds['borderline_low']` and `thresholds['borderline_high']`, but `baseline_tc` only defines `'acceptable'`, `'borderline_high'`, `'high'`. | Add `'borderline_low'` to `baseline_tc` (and align with clinical cutpoints) or use a single fallback range. |
| **Duplicate scenario logic** | Each variable (HDL, TC, glucose, BMI, …) has a block that (1) computes contribution for the **user**, then (2) loops over scenarios and recomputes the same formula. | Use one function per contribution type: `compute_contribution(name, value, context)` where context holds unit_choice, gender, age, etc.; call once for user and once per scenario with scenario values. |

### 2.2 Proposed config structure

- **Single config dict (or YAML/JSON loaded at start)** with sections, e.g.:

```python
CONFIG = {
    "data": {
        "columns_to_ignore": ["DR1TALCO", "DRQSDT11", "DRQSDT12"],
        "numeric_columns": ["LBDHDD", "LBDHDDSI", ...],
        "benchmark_metrics": ["LBDHDD", "LBDHDDSI", ...],
    },
    "missing_codes": {
        "BPQ020": [7, 9], "DPQ010": [7, 9], ...
    },
    "categorical_mappings": {
        "BPQ020": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
        ...
    },
    "numeric_boundaries": {
        "HDL Cholesterol": {"healthy": (50, 70), "unhealthy": (20, 40)},
        ...
    },
    "categorical_adjustments": {
        "High Blood Pressure History": {"healthy": "No", "unhealthy": "Yes"},
        ...
    },
    "utility": {
        "min_utility": 0.1,
        "max_utility": 1.0,
        "benchmark_age": 80,
    },
    "contribution_weights": {
        "hdl_negative": 1.4,
        "tc_negative": 1.3,
        "glucose_negative": 1.3,
        "bmi_negative": 1.2,
        "waist_negative": 1.2,
        "bp_negative": 1.5,
        "diabetes_negative": 1.4,
        ...
    },
    "sigmoid": {
        "k_low": 0.2,
        "k_high": 0.05,
        ...
    },
}
```

- **Improvement:** All tuning (clinical cutpoints, scaling factors, penalty multipliers) becomes editable in one place and can later be validated or estimated from data.

---

## Phase 3: Scoring logic — modular contributions and single utility aggregation

**Goal:** Each domain (HDL, TC, glucose, BMI, waist, BP, diabetes, depression items, PA, diet, etc.) is computed by a **single** function. The main utility score is built by iterating over these contributions and applying caps. No duplicated “user then scenario” blocks; same function is called for user and for each scenario.

### 3.1 Current problems

| Issue | Location / symptom | Correction |
|-------|--------------------|------------|
| **Duplication** | For each variable there is a long block for the **user** and then a loop over scenarios that repeats almost the same formulas (e.g. HDL lines ~402–437 vs ~441–459). | One function per contribution, e.g. `compute_hdl_contribution(hdl_value, unit_choice, gender, config)`. Call it with `(hdl_cholesterol, ...)` for user and with `(scenario_data["HDL Cholesterol"], ...)` for each scenario. |
| **Giant apply over rows** | Lines 2702–2752: one huge `apply` that branches on column name and mixes raw column access with string comparisons that don’t match the data (see Phase 1). | Prefer: (1) build contribution columns using vectorized operations or small per-column functions, or (2) if keeping row-wise logic, ensure the dataframe has the same representation (numeric or string) as the comparison values. Fix the BPQ/DIQ/DRQSDT comparisons. |
| **Contribution list duplicated** | Contribution names and order appear in `contribution_columns`, then again in the manual “After X Contribution” blocks and in the `contributions` dict for the bar chart. | Define the list of contribution names **once**; use it to drive both the aggregation and the chart. |
| **Insulin counted twice** | `contribution_columns` and `filtered_contribution_columns` include `'insulin_contribution'` twice; the apply returns one branch per col so one overwrites the other. | Deduplicate contribution column names; ensure each contribution is computed once. |
| **Order of operations** | Utility is built by a long sequence of `if contribution < 0: utility += contribution * weight` and `utility += contribution`. Order and caps are implicit. | One function: `aggregate_utility(contributions_dict, config) -> float` that applies weights from config, sums, then clamps to `[min_utility, max_utility]`. |

### 3.2 Proposed structure

- **Per-domain functions** (signatures can be refined later):

  - `compute_hdl_contribution(value, unit_choice, gender, benchmarks, config) -> float`
  - `compute_tc_contribution(value, unit_choice, benchmarks, config) -> float`
  - `compute_glucose_contribution(...)`
  - `compute_bmi_contribution(...)`
  - … (one per current “contribution” in the script)

- **Shared helpers** (already partially present; move to top-level or a small `utils` block):

  - `sigmoid(x)` — used in several places; take from config if needed.
  - `age_grouping(age)` — single definition; used for diet, DPQ050, etc.
  - `calculate_dpq_contribution(...)` — keep but ensure it’s called with consistent response coding (numeric 0–3 vs labels).

- **Single aggregation:**

  - `def compute_utility(contributions: dict, config: dict) -> float`
  - Applies per-contribution weights (from config), sums, clamps to `[min_utility, max_utility]`.
  - Main code: build `contributions` dict (user or scenario), then `utility = compute_utility(contributions, CONFIG["utility"])`.

- **QALY:**

  - `def compute_qaly(utility: float, age: int, config: dict) -> float`
  - `remaining_years = max(0, config["benchmark_age"] - age)`; `return round(utility * remaining_years, 2)`.

### 3.3 Correction details for existing logic

- **Categorical comparisons in apply:**  
  Either convert the dataframe used in the apply to labels (same as `nhanes_data_copy`) or change all comparisons to numeric codes (e.g. `row['BPQ020'] == 1` for Yes). Prefer one convention and apply it everywhere.

- **DPQ050 / prevalence_data:**  
  Ensure `prevalence_data` is keyed by **numeric** code (0–3) if the column is numeric after cleaning; and that user input (label like "Nearly every day") is converted to that same code before `.get(age_group, {}).get(dpq050_numeric, 0.0)`.

- **Remove duplicate `insulin_contribution`** from the list of contribution columns so it’s only computed once.

---

## Phase 4: Scenarios — deterministic and consistent

**Goal:** Scenario generation is reproducible and produces plausible combinations. Optionally reduce randomness.

### 4.1 Current problems

| Issue | Location / symptom | Correction |
|-------|--------------------|------------|
| **Fully random** | `random.choice(unhealthy_range)`, `random.randint(...)`, `random.uniform(...)` for best/worst/middle. | Use fixed “typical” values for best/worst (e.g. midpoint of healthy range for best, midpoint of unhealthy for worst) or a fixed seed so runs are reproducible. |
| **middle_case_1 and middle_case_2** | Both set to user value for categoricals; for numerics they get different midpoints. Logic is hard to interpret. | Define explicitly: e.g. “middle_case_1 = average of best and user”, “middle_case_2 = average of worst and user”, or document the intended interpretation. |
| **Unrealistic combinations** | Random draws can combine e.g. very high activity with very low calories. | Optionally add simple consistency checks (e.g. if vigorous_activities == "Yes", days_vigorous_activities >= 1) or clamp scenario values to feasible ranges. |

### 4.2 Improvements

- **Reproducibility:** Set `random.seed(42)` (or configurable) at the start of scenario generation when keeping randomness; or remove randomness and use deterministic “best”/“worst” representatives.
- **Single function:** `generate_scenarios(initial_user_data, config) -> dict[str, dict]` that returns `{"best_case": {...}, "worst_case": {...}, ...}` so the main flow only calls this and then loops over scenarios to compute utilities.

---

## Phase 5: Output and visualization — separate from calculation

**Goal:** All `print` and `plt` calls are in one place; the scoring pipeline returns structured results (dict with utility, qaly, contributions, warnings) and optionally a “report” string. Colab cell can then print and plot from that result.

### 5.1 Current problems

| Issue | Location / symptom | Correction |
|-------|--------------------|------------|
| **Prints and plots inside scoring** | Dozens of `print(f"After X Contribution: ...")` and later `plt.figure`, `plt.show()` mixed with calculation. | Scoring functions return values only; a single `format_report(result)` (or similar) produces the text; a single `plot_contributions(result)` (or `plot_scenario_comparison(...)`) produces the chart(s). |
| **Warnings list** | Built as a side effect inside the long function. | Append to a `warnings` list that is part of the returned result dict; main code or report function prints them. |
| **Bar chart data** | The `contributions` dict for the chart is rebuilt after the fact (lines 2930–2958); risk of drift from actual aggregation. | Build the contributions dict once (as in Phase 3); use that same dict for both aggregation and plotting. |

### 5.2 Proposed flow

- **Pipeline returns:**

  ```python
  result = {
      "utility_score": float,
      "qaly": float,
      "remaining_years": int,
      "contributions": {"HDL": 0.02, "Total Cholesterol": -0.01, ...},
      "warnings": ["Warning: ..."],
      "scenario_qaly_scores": {"best_case": 42.5, "worst_case": 28.1, ...},
  }
  ```

- **Single place at the end of `qaly()`:**

  - Call a function that runs: load data → build benchmarks → compute user contributions → aggregate utility → compute QALY → generate scenarios → compute scenario utilities/QALYs.
  - Assign result to `result`.
  - Then: `print(format_report(result))`, and optionally `plot_contributions(result)` / `plot_scenario_comparison(result)`.

---

## Phase 6: Entry point and execution

**Goal:** Colab still runs the same cell with `qaly()` at the bottom; internally `qaly()` is a short orchestrator that collects @param-style inputs (or in Colab they remain as widget-bound variables), calls the new modules, and then prints/plots.

### 6.1 Current

- One giant `qaly()` that does everything; `qaly()` at the bottom runs on import/run.

### 6.2 After refactor

- **Option A (minimal change):**  
  `qaly()` stays as the only entry point. It:
  1. Builds `initial_user_data` from the current variables (gender, age, etc. — still coming from @param in Colab).
  2. Calls `load_and_clean_nhanes(...)`.
  3. Builds benchmarks from the cleaned data.
  4. Calls `generate_scenarios(initial_user_data, CONFIG)`.
  5. For user: builds contribution dict → `compute_utility` → `compute_qaly` → store in `result`.
  6. For each scenario: same with scenario data → store in `result["scenario_qaly_scores"]`.
  7. Appends any warnings to `result["warnings"]`.
  8. Calls `print(format_report(result))` and `plot_contributions(result)` (and scenario plots if desired).

- **Option B (slightly more structure):**  
  Split into two cells in Colab:  
  - Cell 1: @param widgets and `initial_user_data = {...}`.  
  - Cell 2: `result = run_qaly_pipeline(initial_user_data, data_path=file_path)` then `print(format_report(result))` and `plot_contributions(result)`.  
  Then `run_qaly_pipeline` is a function that takes inputs and path and returns the result dict; `qaly()` can simply be `result = run_qaly_pipeline(initial_user_data)`; `print(...)`; `plot_...(result)`.

- **Execution on import:**  
  Keep `if __name__ == "__main__": qaly()` or equivalent so that when the script is **imported** (e.g. for tests or future reuse), it doesn’t run the full pipeline; when run as a script or in Colab, it does. In Colab, the “run cell” semantics already avoid import-side execution if you don’t import the file; so the main fix is to avoid running `qaly()` at module level. Prefer:

  ```python
  if __name__ == "__main__":
      qaly()
  ```

  and in Colab, either run the script or keep the code in a notebook where the last cell calls `qaly()`.

---

## Implementation order (recommended)

1. **Phase 1** — Data loader and cleaning; fix categorical representation so the apply (or its replacement) sees consistent data. Fix the BPQ/DIQ string-vs-numeric bug.
2. **Phase 2** — Extract CONFIG (and fix `baseline_tc` missing key); no change to control flow yet.
3. **Phase 3** — Extract one contribution function (e.g. HDL), use it for both user and scenarios; then replicate the pattern for the others; add `compute_utility` and `compute_qaly`; deduplicate `insulin_contribution`.
4. **Phase 4** — Refactor scenario generation into `generate_scenarios(...)` and add seed or deterministic values.
5. **Phase 5** — Introduce result dict, `format_report`, and `plot_contributions`; remove prints from inside contribution logic.
6. **Phase 6** — Shorten `qaly()` to the orchestrator; add `if __name__ == "__main__": qaly()`.

---

## File layout (optional, for local / tests)

If you later split out of a single notebook/script:

- `qaly_config.py` — CONFIG and categorical_mappings.
- `qaly_data.py` — load_and_clean_nhanes, build benchmarks.
- `qaly_contributions.py` — compute_*_contribution, compute_utility, compute_qaly.
- `qaly_scenarios.py` — generate_scenarios.
- `qaly_report.py` — format_report, plot_contributions.
- `ver_3_qaly_nhanes_calculation.py` (or Colab notebook) — @param / widgets, then call the above and print/plot.

For now, all of this can live in the same file with clear section comments (e.g. `# --- Config ---`, `# --- Data loading ---`, `# --- Contribution functions ---`) so that later extraction to modules is straightforward.

---

## Summary of corrections and improvements

| Area | Correction | Improvement |
|------|------------|------------|
| **Data** | Use one cleaned dataframe for both benchmarks and contribution apply; fix categorical representation (string vs numeric). | Single loader; explicit dtypes; optional local path. |
| **Config** | Fix `baseline_tc` missing key. | All weights/boundaries/mappings in CONFIG. |
| **Scoring** | Fix duplicate insulin column; fix categorical comparisons in apply. | One function per contribution; single aggregation and QALY functions; same code path for user and scenarios. |
| **Scenarios** | N/A (logic can stay). | Deterministic or seeded; single `generate_scenarios()`. |
| **Output** | N/A | Result dict; format_report and plot_contributions; no prints inside scoring. |
| **Entry** | N/A | `qaly()` as short orchestrator; `if __name__ == "__main__": qaly()`. |

This plan keeps your Colab + @param workflow intact while making the code correct, maintainable, and ready for incremental improvement (e.g. config-driven weights or validated utility tariffs later).
