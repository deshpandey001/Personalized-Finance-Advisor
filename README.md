# AI Portfolio Advisor

An AI-powered personal finance and portfolio advisor built with Flask, classical machine learning, and an LLM (Gemini) for human-friendly explanations.

The app takes a user’s basic financial profile (age, income, savings, risk score, and goals), generates a stock/bond/cash allocation, applies simple compliance rules, simulates long‑term growth, and uses an LLM to explain the recommendation in plain language.

---

## ✨ Features

- **Interactive web UI**
  - Landing page with a financial profile form
  - Results dashboard with cards and charts (allocation, projections, risk stats, insurance)

- **ML-driven portfolio allocation**
  - Trained portfolio model (Random Forest) on synthetic data
  - Predicts allocation across **stocks, bonds, cash**
  - Simple risk/return analytics

- **LLM explanations (Gemini)**
  - Natural language explanation of:
    - Why the allocation matches the user’s profile
    - Key risk considerations
    - High-level rationale
  - Extracts **life events** from free-form goals text and uses them in the explanation

- **Compliance guardrails (simplified)**
  - Example rule: cap equity allocation for low risk scores
  - Provide user-facing explanation of any adjustments

- **Insurance recommendations (rule-based)**
  - Basic suggestions based on age, income, and family status (mocked flags)

---

## 🏗 Architecture Overview

**Stack**

- **Backend / Web framework:** Flask
- **ML / Data:** pandas, numpy, scikit-learn
- **LLM:** Google Gemini (via custom `initialize_gemini` / `get_llm_explanation` in `finance.py`)
- **Environment management:** python-dotenv
- **Frontend:** Jinja2 templates (`templates/`), CSS (`static/css/main.css`)

**High-level flow**

1. User opens `/` → sees HTML form.
2. On submit (`POST /`):
   - Parse form data (age, income, savings, risk_score, goals).
   - Build a `pandas.DataFrame` with numeric features.
   - Run the **portfolio model** to predict `[stocks, bonds, cash]`.
   - Extract top features from the model for context.
   - Use Gemini to:
     - Extract **life events** from `goals` text.
     - Generate a natural language explanation of the allocation.
   - Apply **compliance rules** to adjust allocation if needed.
   - Generate growth projections and simple risk/return stats.
   - Get **insurance recommendations**.
   - Render `results.html` with all of the above.

3. Optional programmatic access via `/predict` (JSON API).

---

## 📁 Project Structure

> Paths are relative to the repo root.

```text
pbl/
├─ app.py                  # Flask app: routes, orchestration
├─ finance.py              # Core finance / ML / LLM logic (imported by app.py)
├─ requirements.txt        # Python dependencies (recommended)
├─ .env                    # Environment variables (not committed)
├─ templates/
│  ├─ index.html           # Input form page
│  └─ results.html         # Results dashboard page
└─ static/
   └─ css/
      └─ main.css          # Styling for index + results
```

> Note: `finance.py` is referenced in `app.py` and should implement:
> - `train_portfolio_model`
> - `initialize_gemini`
> - `get_llm_explanation`
> - `apply_compliance_rules`
> - `get_insurance_recommendation`
> - `generate_synthetic_data`
> - `extract_life_events`

---

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.9+ (recommended)
- Git
- A Gemini API key (or comment out LLM parts for offline use)

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

On your machine this corresponds to:

```bash
cd "c:\Users\Varad Deshpande\OneDrive\Desktop\pbl"
```

### 3. Create and activate a virtual environment (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 4. Install dependencies

If you have `requirements.txt`:

```powershell
pip install -r requirements.txt
```

If not, minimally:

```powershell
pip install flask pandas numpy scikit-learn python-dotenv
# plus any Gemini client / Google libraries required by finance.py
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add (example; adjust names to match your `finance.py` implementation):

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
# Any other custom config, e.g.:
# GEMINI_MODEL_NAME=gemini-1.5-flash
# FLASK_ENV=development
```

`app.py` calls `load_dotenv()`, so these will be loaded automatically.

---

## 🚀 Running the Application

From the project root, with your virtual environment active:

```powershell
python app.py
```

You should see output like:

```text
Initializing models and LLM client...
Initialization complete.
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Then open your browser at:

- **Frontend (UI):** http://127.0.0.1:5000/

### Flow

1. Fill in:
   - Age
   - Income
   - Savings
   - Risk score (slider or input)
   - Life goals (free text)
2. Submit
3. You’ll see:
   - Recommended allocation (stocks/bonds/cash)
   - Expected portfolio return & “risk” (simplified)
   - Growth projections (10/20/30 year)
   - LLM explanation card
   - Insurance recommendations
   - Any extracted life events integrated into the explanation

---

## 🌐 API Endpoints

### `GET /`

Renders the input form (`index.html`).

### `POST /`

Processes the HTML form and renders the results dashboard (`results.html`).

- **Request:** form-encoded fields:
  - `age` (int)
  - `income` (int)
  - `savings` (int)
  - `risk_score` (float)
  - `goals` (string, optional)

- **Response:** HTML page showing:
  - `allocation` (stocks/bonds/cash)
  - `analysis` (expected_return, portfolio_risk, per-asset stats)
  - `gemini_explanation` (LLM)
  - `projections` (growth trajectories)
  - `insurance_recommendations`
  - `compliance_explanation`
  - `life_events` (if extracted)

### `POST /predict` (JSON API)

Programmatic endpoint that returns predictions and explanations as JSON.

**Request body (JSON):**

```json
{
  "age": 30,
  "income": 60000,
  "savings": 100000,
  "risk_score": 6.5
}
```

**Response (JSON):**

```json
{
  "predicted_allocation": {
    "stocks": 60.0,
    "bonds": 30.0,
    "cash": 10.0
  },
  "llm_explanation": "Text explanation or fallback message",
  "adjusted_allocation": {
    "stocks": 55.0,
    "bonds": 35.0,
    "cash": 10.0
  },
  "compliance_explanation": "Stocks capped at 55% due to your risk profile.",
  "insurance_recommendations": [
    "Example recommendation 1",
    "Example recommendation 2"
  ]
}
```

> **Note:** The exact values depend on your `finance.py` logic.

---

## 🧠 ML & LLM Components (High-Level)

- **Synthetic data generation**  
  `generate_synthetic_data()` creates a training dataset for the portfolio model.

- **Model training**  
  `train_portfolio_model(df)`:
  - Prepares features and labels.
  - Trains a model (e.g., RandomForestRegressor).
  - Returns the trained model and feature names.
  - This runs **once at app startup**.

- **Inference**
  - For each request, a `pandas.DataFrame` is created from:
    - `age`, `income`, `savings`, `risk_score`.
  - The model predicts `[stocks, bonds, cash]`.
  - Feature importances are aggregated across estimators to get top drivers.

- **LLM Explanation (Gemini)**
  - `initialize_gemini()` sets up the Gemini client using the API key from `.env`.
  - `extract_life_events(gemini_model, goals_text)` parses goals into structured events.
  - `get_llm_explanation(gemini_model, user_profile, prediction, top_features_names, life_events)` builds a structured prompt and gets a natural language rationale.

---

## 🔐 Security & Privacy (Basic)

- API keys are loaded via `.env` (not committed).
- Input is validated and converted to numeric types in `app.py`.
- LLM is used only for **explanations**, not for making final numeric decisions.
- This project is meant for **educational and prototyping** purposes, not production‑grade investment advice.

---

## 📦 Deployment (Overview)

Some ideas for deployment:

- **Containerization:**  
  - Create a `Dockerfile` that:
    - Installs Python + dependencies
    - Copies `app.py`, `finance.py`, `templates`, `static`
    - Exposes port 5000 and runs `gunicorn app:app`
- **Cloud options:**
  - AWS ECS/Fargate, Azure App Service, or GCP Cloud Run.
  - Use a managed DB (if you add persistence) and a secret manager for the Gemini key.
- **Static assets:**
  - Served via the Flask `static/` folder or offloaded to a CDN in production.

---

## 🧪 Development & Testing

Recommended practices (you can extend):

- Add unit tests for:
  - `train_portfolio_model` (shape, ranges).
  - `apply_compliance_rules` (correct caps).
  - Dummy tests for `get_insurance_recommendation`.
- Add integration tests that:
  - Call `/predict` with sample payloads.
  - Assert expected keys and value ranges in the response.

Run tests (if using `pytest`):

```bash
pytest
```

---

## 📝 Disclaimer

This project is for **learning, experimentation, and demonstration** only.

It does **not** provide regulated financial advice, and any outputs are purely illustrative. Do not use this system to make real investment decisions without consulting a qualified professional.

---

## 📄 License

Specify your license here, e.g.:

```text
MIT License
```

(Or add a `LICENSE` file and reference it.)
