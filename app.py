"""
Gradio demo app — HF Spaces compatible (place this file at repo root).

In local mode it hits the FastAPI server at API_URL.
For HF Spaces, set the HF_API_URL secret or update API_URL to your deployed endpoint.

Start locally:
    # Terminal 1: uv run uvicorn serving.api:app --port 8000
    # Terminal 2: uv run python app.py
"""
from __future__ import annotations

import os

import gradio as gr
import requests

API_URL = os.getenv("HF_API_URL", "https://huggingface.co/spaces/madchavez/MLOps_end2end_MSDS2026:8000")

PRIMARY_USAGE_MAP = {
    "Work / Productivity (1)": 1,
    "Gaming (2)": 2,
    "Study / Education (3)": 3,
    "Personal / Entertainment (4)": 4,
}
BRAND_MAP = {
    "Brand A (1)": 1,
    "Brand B (2)": 2,
    "Brand C (3)": 3,
    "Brand D (4)": 4,
    "Brand E (5)": 5,
}


def predict(
    hours_per_day: float,
    cost: float,
    user_age: float,
    primary_usage: str,
    brand: str,
    computer_age_months: float,
) -> tuple[str, str]:
    payload = {
        "hours_used_per_day": hours_per_day,
        "cost": cost,
        "user_age": user_age,
        "primary_usage": PRIMARY_USAGE_MAP[primary_usage],
        "brand": BRAND_MAP[brand],
        "computer_age_months": computer_age_months,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return (
            "❌ Cannot connect to the prediction API.",
            f"Make sure the FastAPI server is running at {API_URL}\n\n"
            "Run: uv run uvicorn serving.api:app --port 8000",
        )
    except Exception as exc:
        return "❌ Error", str(exc)

    prob = data["probability"]
    label = data["needs_replacement"]
    version = data.get("model_version", "N/A")

    verdict = "🔴 **Replacement Recommended**" if label else "🟢 **No Replacement Needed**"
    details = (
        f"**Replacement Probability:** {prob:.1%}\n\n"
        f"**Decision Threshold:** 50%\n\n"
        f"**Model:** {version}\n\n"
        f"---\n"
        f"*Inputs:* {hours_per_day:.1f}h/day · ${cost:,.0f} · "
        f"{user_age:.0f}yo · {computer_age_months:.0f} months old"
    )
    return verdict, details


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Computer Durability Predictor",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# 💻 Computer Durability Predictor
**MLOps End-to-End Pipeline** · Dagster · MLflow · Evidently · FastAPI · Gradio

Predict whether a computer **needs replacement** based on its usage profile.
The model was trained on the Computer Durability dataset using
**XGBoost + Optuna** hyperparameter tuning and tracked with **MLflow**.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Usage & Hardware")
                hours = gr.Slider(
                    label="Hours Used Per Day",
                    minimum=1.0, maximum=24.0, step=0.5, value=10.0,
                )
                cost = gr.Slider(
                    label="Computer Cost (USD)",
                    minimum=5000, maximum=50000, step=500, value=30000,
                )
                comp_age = gr.Slider(
                    label="Computer Age (Months)",
                    minimum=1, maximum=60, step=1, value=24,
                )

            with gr.Column(scale=1):
                gr.Markdown("### User Profile")
                user_age = gr.Slider(
                    label="User Age (Years)",
                    minimum=8, maximum=65, step=1, value=35,
                )
                usage_type = gr.Dropdown(
                    label="Primary Usage",
                    choices=list(PRIMARY_USAGE_MAP.keys()),
                    value="Work / Productivity (1)",
                )
                brand = gr.Dropdown(
                    label="Brand",
                    choices=list(BRAND_MAP.keys()),
                    value="Brand C (3)",
                )

        predict_btn = gr.Button("Predict", variant="primary", size="lg")

        with gr.Row():
            verdict_out = gr.Markdown(label="Verdict")
            details_out = gr.Markdown(label="Details")

        predict_btn.click(
            fn=predict,
            inputs=[hours, cost, user_age, usage_type, brand, comp_age],
            outputs=[verdict_out, details_out],
        )

        gr.Markdown(
            """
---
### Architecture
```
CSV Data → Dagster Pipeline → MLflow (tracking + registry)
                           → Evidently (drift + quality reports)
                           → FastAPI  → Gradio (this app)
```
**Key signals driving replacement probability:**
- Hours/day > 12 → risk increases significantly
- Cost < $20k → cheaper machines fail more often
- Older users (>40) → slight additional risk
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
