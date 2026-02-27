# QALY 2024 — NHANES-based population comparator

Local web app (HTML, CSS, JS, YAML) to run the QALY/NHANES-style calculator. You can run it locally and publish the frontend to a GitHub repo.

## What’s in this repo

- **`web/`** — Frontend
  - **`index.html`** — Single-page form (demographics, labs, diet, mental health, activity)
  - **`css/style.css`** — Styles
  - **`js/app.js`** — Sends form data to `/api/calculate`, shows utility, QALY, warnings, and two charts
  - **`config.yaml`** — Config (app title, utility bounds, etc.); loaded by the frontend
- **`server.py`** — Flask app: serves `web/` and `POST /api/calculate` (runs the Python script, returns JSON)
- **`ver_3_qaly_nhanes_calculation.py`** — Full QALY calculation; callable as `qaly(initial_user_data=..., data_path=...)` for the API
- **`Cleaned_Dataset_QALY_Diet.csv`** — NHANES-derived dataset (required for the calculator)
- **`REFACTOR_PLAN.md`** — Refactor and improvement plan for the Python script

## Run locally (with real calculation and graphs)

1. Install Python dependencies: `pip install -r requirements.txt`
2. Start the backend: `python server.py`
3. Open **http://127.0.0.1:5000** in your browser. Fill the form and click **Calculate**. You get utility score, QALY, warnings, a **contribution breakdown** chart (horizontal bars), and a **scenario comparison** chart (QALY for best/worst/middle scenarios).

<details>
<summary>Alternative: static-only (no backend)</summary>

### Option 1: Node (npx)

```bash
cd web
npx serve .
```

Then open **http://localhost:3000** (or the URL shown).

### Option 2: Python

```bash
cd web
python3 -m http.server 8000
```

Then open **http://localhost:8000**.

### Option 3: VS Code / Cursor

Use the “Live Server” extension and “Open with Live Server” on `index.html`.

---

Without the Flask backend, **Calculate** will show an error; run `python server.py` for full results and charts.

## Put the app on GitHub

1. Create a new repository on GitHub (e.g. `qaly-nhanes-app`).
2. From the project root (`QALY_2024`):

   ```bash
   git init
   git add .
   git commit -m "Initial commit: web app + Python ref"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

3. To **publish the web app** with GitHub Pages:
   - Repo **Settings → Pages**
   - Source: **Deploy from a branch**
   - Branch: **main**, folder: **/ (root)** or **/web** depending on where your `index.html` lives.
   - If you choose root, move `index.html` (and `css/`, `js/`, `config.yaml`) to the repo root, or set the Pages source to **main** and **/web** so the site is served from the `web` folder.

**Note:** GitHub Pages serves static files only. The current app is static; if you add a Python backend later, you’ll need to host it elsewhere (e.g. Railway, Render, or a VPS) and point the frontend to that API.

## Config (YAML)

Edit **`web/config.yaml`** to change:

- `app.title` and `app.subtitle`
- `utility.min_score`, `max_score`, `benchmark_age`
- `weights.*` (penalty multipliers)
- `numeric_boundaries` (healthy/unhealthy ranges for validation or scenarios)
- `options` (dropdown choices; can be used to build or validate the form)

The app loads this file when the page loads. After changing YAML, refresh the page.

## License and disclaimer

For research / educational use. Not a substitute for professional medical advice.
