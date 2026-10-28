# SupportScope AI — Grounded Customer Support Assistant

SupportScope AI is a small Generative AI application for answering customer-support questions using a fixed policy knowledge base for a fictional store, Northstar Store. It demonstrates a defined assistant remit, deterministic guardrails, retrieval-grounded responses, repeatable evaluation, and token/cost telemetry.

> **Demo scope:** Northstar Store and its policies are fictional. The assistant cannot access or modify real orders, payments, refunds, or customer accounts.

## Features

- **Defined remit:** shipping, returns, refunds, cancellations, damaged items, warranties, payment methods, and store contact information.
- **Policy-grounded answers:** TF-IDF retrieval selects up to two relevant policy excerpts from a version-controlled JSON knowledge base.
- **Guardrails:** known unsafe request patterns are refused before a model call. Requests without relevant policy evidence are handled as out of scope.
- **Source IDs:** the application attaches citations corresponding to retrieved policy excerpts.
- **Evaluation:** 22 JSONL test cases cover supported questions, off-topic requests, and unsafe requests. A scoring script records expected status, required facts, and source checks.
- **Usage telemetry:** input/output/total tokens, request latency, retrieval score, and an estimated reference cost are shown for model responses.

## Screenshots

| Grounded policy response and session usage | Out-of-scope refusal and session usage |
|---|---|
| ![SupportScope AI answering a store policy question with source IDs and usage metrics](Screenshots/Screenshot%202026-10-28%20203229.png) | ![SupportScope AI refusing an off-topic question while showing session token and cost metrics](Screenshots/Screenshot%202026-10-28%20203306.png) |

## Architecture

```text
User question
    ↓
Input validation and unsafe-pattern guardrail
    ├── Unsafe / invalid ──→ Refusal without a model call
    ↓
TF-IDF retrieval over fixed store policies
    ├── No relevant evidence ──→ Out-of-scope refusal
    ↓
Top two policy excerpts + restricted prompt
    ↓
Groq-hosted openai/gpt-oss-20b
    ↓
Answer + application-attached source IDs
    ↓
Token usage, estimated cost, retrieval score, and latency
```

## Technology stack

| Component | Technology | Purpose |
|---|---|---|
| Language and UI | Python 3.13, Streamlit | Application logic and chat interface |
| Hosted language model | Groq API, `openai/gpt-oss-20b` | Generate answers grounded in retrieved evidence |
| Retrieval | scikit-learn TF-IDF | Lightweight lexical retrieval without a paid embedding service |
| Knowledge base | JSON | Fixed, version-controlled fictional store policies |
| Evaluation | JSONL, Python scoring script, pytest | Repeatable evaluation and automated unit tests |
| Configuration | `python-dotenv` | Load local environment variables |

## Evaluation results

Recorded benchmark run: **28 October 2026**. The evaluation set contains 22 cases. A case passes when its expected status, required factual checks, and expected source checks pass under the project's explicit scoring rubric.

| Expected behavior | Passed | Pass rate |
|---|---:|---:|
| Supported policy questions | 9 / 14 | 64.3% |
| Off-topic refusals | 4 / 4 | 100% |
| Unsafe-request refusals | 4 / 4 | 100% |
| **Overall** | **17 / 22** | **77.3%** |

| Usage / performance measure | Recorded result |
|---|---:|
| Input tokens | 6,432 |
| Output tokens | 2,108 |
| **Total tokens** | **8,540** |
| Estimated reference cost | **$0.00111479 USD** |
| Mean latency | 1.094 seconds |
| P50 latency | 1.253 seconds |
| Source check rate | 100% |
| Automated unit tests | **9 passed** (`pytest`, 4.47 seconds) |

The 77.3% figure is the pass rate for this recorded benchmark and rubric, not a general claim of semantic accuracy. Five cases failed in the recorded run, all among supported policy questions. The case-level results are included in `reports/baseline_evaluation_results.csv`; the summary is in `reports/baseline_evaluation_summary.md`. Model outputs and latency can vary between runs, so failed cases should be reviewed before comparing future scores.

Cost is calculated from configurable per-million-token reference rates in `.env`. It is an estimate for comparison, not a billing statement. Free-plan usage is subject to provider limits. Requests refused before model invocation use zero model tokens.

## Run locally

Python 3.13 is recommended. In PowerShell, run these commands from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env` using a key from <https://console.groq.com/keys>. The model name and reference input/output rates are configurable in the same file. Keep `.env` private and never commit API keys.

Run the tests and application:

```powershell
pytest
streamlit run app.py
```

Run the benchmark to regenerate case-level and aggregate reports:

```powershell
python scripts/evaluate.py
```

The evaluator writes `evaluation_results.json`, `evaluation_results.csv`, and `evaluation_summary.md` under `reports/`. The version-controlled `baseline_evaluation_*` files preserve the results of the recorded benchmark run.

## Scope and limitations

SupportScope AI is a portfolio demonstration of narrow-scope assistance, evidence retrieval, guardrails, evaluation, and usage measurement. It does not integrate with a real order-management system and has no authentication, database, or persistent customer history. The safety filter uses defined patterns and cannot guarantee detection of every unsafe or obfuscated request. TF-IDF is lexical retrieval, and citations indicate which passages were retrieved—not proof that every generated sentence is entailed by those passages.

The application is designed to run without a local GPU. The language model is hosted through Groq, and the interface can be deployed through Streamlit Community Cloud. Free-tier availability, quotas, and model access may change.

## Author

Arjun Goel

MCA'27 @VIT Vellore
