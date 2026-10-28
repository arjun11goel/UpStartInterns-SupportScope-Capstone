# Evaluation reports

Run `python scripts/evaluate.py` after configuring `GROQ_API_KEY` to create:

- `evaluation_results.json` — summary, full answers, per-case scores, token counts, latency, and estimated cost.
- `evaluation_results.csv` — case-level scores for spreadsheet inspection.
- `evaluation_summary.md` — compact Markdown report.

These generated files are ignored by Git by default because results can contain model outputs and change on each run. After you run the evaluation, inspect the outputs and intentionally add the summary/report you want your reviewer to see. Never put API keys or private customer data in a report.
