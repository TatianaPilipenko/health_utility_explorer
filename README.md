## Health utility explorer — NHANES-based population comparator

Local, explainable **health utility explorer** built on NHANES-style data.

The app:

- collects a set of **demographic, lab, diet, mental health, and activity** inputs,
- compares them to a **NHANES-derived dataset**, and
- computes a **utility-style health score** and **estimated quality‑adjusted years** (to a benchmark age),
  with detailed **factor contributions** and **scenario comparisons**.

It is designed for **exploration and education**, not for clinical or reimbursement decisions.

---

### Key features

- **Single-page web app** (`web/`) with a clean form UI.
- **Utility score** (0–1) plus **estimated quality‑adjusted years** to a configurable benchmark age.
- **Contribution breakdown**: which factors raise or lower the utility score (green vs red bars).
- **Cumulative progression**: how the score evolves as each factor is applied.
- **Scenario explorer** (best / worst / middle‑case profiles):
  - Cumulative utility by scenario (multiple lines).
  - Scenario comparison of contributions (grouped tornado chart).
- Fully local: **Flask backend + JavaScript frontend**, no external APIs.

---

## Project structure

- **`server.py`**  
  Flask backend:
  - serves the static frontend from `web/`
  - exposes `POST /api/calculate` that runs the calculator and returns JSON

- **`qaly_calculator.py`**  
  Main **calculation engine** used by the web app. This is the file to modify if you change the logic.

- **`ver_3_qaly_nhanes_calculation.py`**  
  Original Colab-style script kept as a **reference**. The web app uses `qaly_calculator.py`, not this file.

- **`Cleaned_Dataset_QALY_Diet.csv`**  
  NHANES‑derived dataset used by the calculator.  
  - Required at runtime by `qaly_calculator.py` / `server.py`.  
  - You may choose **not** to commit this file to GitHub (e.g. add it to `.gitignore`) and instead document how to obtain it.

- **`web/`** — frontend
  - **`index.html`** – single-page UI (form + result panels).
  - **`css/style.css`** – layout and styling.
  - **`js/app.js`** – loads config, handles form submission, calls `/api/calculate`, renders results and charts.
  - **`config.yaml`** – app title/subtitle, utility bounds, contribution weights, and options.

- **`REFACTOR_PLAN.md`**  
  Notes about how the original script was reshaped into a reusable calculator and web backend.

- **`requirements.txt`**  
  Python dependencies for the backend and calculation.

---

## Installation and setup

### 1. Create / activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

From the project root (`QALY_2024`):

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Make sure `Cleaned_Dataset_QALY_Diet.csv` is present in the **project root**, next to `server.py` and `qaly_calculator.py`.

If you do **not** want to commit the dataset to GitHub, add it to `.gitignore` and describe in this README (or your paper/report) how to obtain or regenerate it.

---

## Running the Health utility explorer locally

From the project root:

```bash
python server.py
```

Then open:

```text
http://127.0.0.1:5000
```

in your browser.

### What you’ll see

1. **Form** with sections:
   - Demographics
   - Labs and clinical measures
   - Diet and nutrients
   - Mental health and functioning
   - Physical activity and sedentary time

2. After clicking **Calculate**, the app:
   - POSTs your inputs to `/api/calculate`.
   - Shows:
     - **Utility score** (0–1).
     - **Remaining years to benchmark age** (e.g. to age 80).
     - **Estimated quality‑adjusted years**.
     - Any **notes/warnings** (missing data, caps, etc.).

3. It also renders several charts:
   - **Utility score contribution breakdown**  
     Horizontal bars. **Green = raises your score, red = lowers it.**
   - **Cumulative utility score progression**  
     How the utility score changes as each factor is applied in sequence.
   - **Cumulative utility by scenario**  
     One line per scenario (Best case, Middle cases, Worst case).
   - **Scenario comparison (contributions)**  
     Grouped horizontal bars showing how the same parameters behave under each scenario.

If the browser shows “Failed to reach backend” or similar, ensure that:

- `python server.py` is running, and
- you are opening the app via `http://127.0.0.1:5000` (the Flask server), not via a file URL.

---

## Interpreting the numbers

- **Utility score (0–1)**  
  - 1.0 ≈ ideal health state (in this model).  
  - Closer to 0.1 ≈ poorer health state (minimum cap used in the model).

- **Remaining years to benchmark age**  
  - Calculated as `benchmark_age - current_age` (e.g. up to age 80).
  - This is **not** a life expectancy prediction; it is just a horizon used to scale quality‑adjusted years.

- **Estimated quality‑adjusted years**  
  - Roughly `utility_score × remaining_years` using the benchmark age horizon.
  - Provided for **exploratory comparison only**.

- **Scenarios**  
  - **Best case**: favorable values drawn from healthier parts of the NHANES‑style distribution.  
  - **Worst case**: unfavorable values.  
  - **Middle cases**: intermediate mixes to illustrate sensitivity.

The calculations are driven by `qaly_calculator.py` and are **model‑based**, not clinical guidance.

---

## Configuration (`web/config.yaml`)

You can tune some high‑level behavior without touching Python:

- **`app.title` / `app.subtitle`** – text in the page header.
- **`utility.min_score` / `max_score` / `benchmark_age`** – clamps and horizon used in the model.
- **`weights.*`** – multipliers for negative contributions (how strongly to penalize certain risk factors).
- **`numeric_boundaries`** – approximate “healthy”/“unhealthy” ranges used for validation and scenarios.
- **`options`** – choices for dropdowns and radios (gender, depression frequency, diet type, etc.).

After editing `config.yaml`, simply refresh the page in your browser.

---

## Development notes

- **Where to change the logic:**  
  - Use `qaly_calculator.py`. The web app calls `qaly(initial_user_data=..., data_path=...)` from this module.
  - `ver_3_qaly_nhanes_calculation.py` is kept as a historical reference and is not used by `server.py`.

- **Frontend behavior:**  
  - `web/js/app.js` loads `config.yaml`, serializes the form, calls `/api/calculate`, and renders:
    - main numeric results,
    - warnings,
    - several Chart.js visualizations.

- **Backend plotting:**  
  - `server.py` forces a non‑GUI matplotlib backend (`Agg`) so the calculation logic can still generate plots without opening windows (important on macOS / threaded environments).

---

## License and disclaimer

This code is intended for **research and educational use only**.

It is **not** a medical device and **must not** be used to diagnose, treat, or make decisions about individual patients or reimbursement. Always consult qualified health professionals for medical advice.
