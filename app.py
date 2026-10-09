import html
import os

import gradio as gr
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rf_model = joblib.load(os.path.join(BASE_DIR, "disease_predictor_rf.pkl"))
mlb = joblib.load(os.path.join(BASE_DIR, "symptom_encoder.pkl"))
symptom_choices = sorted(list(mlb.classes_))

custom_css = """
.gradio-container { max-width: 1100px !important; margin: auto !important; }
#main-header { text-align: center; padding: 25px 12px; }
#main-header h1 { font-size: 38px; margin-bottom: 8px; }
#main-header p { font-size: 16px; opacity: 0.8; }
#predict-btn { border-radius: 12px !important; font-weight: 700 !important; min-height: 48px; }
#results { border-radius: 16px; padding: 16px; }
"""

def predict_health(selected_symptoms):
    if not selected_symptoms:
        return "## 🩺 Select your symptoms\n\nChoose one or more symptoms and click **Analyze Symptoms**."

    known_symptoms = set(mlb.classes_)
    selected = list(dict.fromkeys(s for s in selected_symptoms if s in known_symptoms))
    if not selected:
        return "No recognized symptoms found. Please select from the list."

    features = mlb.transform([selected])
    probabilities = rf_model.predict_proba(features)[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]

    result = "# 🩺 Analysis Results\n\nThese are the model's three highest-ranked predictions.\n\n### Symptoms selected\n\n"
    result += " ".join(f"`{html.escape(s.replace('_', ' ').title())}`" for s in selected)
    result += "\n\n---\n\n### 🔎 Top 3 model predictions\n\n"

    for rank, idx in enumerate(top_indices, start=1):
        disease = html.escape(str(rf_model.classes_[idx]))
        score = float(probabilities[idx]) * 100
        bar_width = max(0, min(100, score))
        result += f"""
<div style="border:1px solid var(--border-color-primary); border-radius:14px; padding:16px; margin:12px 0;">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
    <div><div style="font-size:12px; opacity:.7;">RANK {rank}</div>
    <div style="font-size:18px; font-weight:650;">{disease}</div></div>
    <div style="font-size:18px; font-weight:700;">{score:.1f}%</div>
  </div>
  <div style="background:var(--border-color-primary); border-radius:8px; height:8px; margin-top:14px;">
    <div style="background:#0D9488; width:{bar_width:.2f}%; height:8px; border-radius:8px;"></div>
  </div>
  <div style="font-size:11px; opacity:.7; margin-top:6px;">Relative model score, not a medical probability</div>
</div>
"""

    result += """
---
### ⚠️ Medical disclaimer

This is an educational prototype, not a clinically validated diagnostic tool.
These predictions cannot confirm or rule out a disease. Do not use the scores
to choose medication or treatment. Consult a qualified healthcare professional
for medical concerns.
"""
    return result

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="emerald"),
    css=custom_css,
    title="HealthAI | AI Health Assistant"
) as demo:
    gr.HTML("""
    <div id="main-header">
      <div style="font-size:42px;">💚</div>
      <h1>HealthAI</h1>
      <p>Your AI-powered health learning assistant</p>
      <p>Explore symptom patterns using a machine learning model.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("## 🩺 How are you feeling today?")
            gr.Markdown("Search for symptoms and select all that apply.")
            symptom_picker = gr.Dropdown(
                choices=symptom_choices,
                multiselect=True,
                filterable=True,
                label="Search & select symptoms"
            )
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze Symptoms", variant="primary", elem_id="predict-btn")
                clear_btn = gr.Button("✕ Clear", variant="secondary")
            gr.Markdown("**Tip:** Only symptoms in the dataset can be selected.")

        with gr.Column(scale=6):
            results = gr.Markdown(
                value="## Your results will appear here\n\nSelect symptoms and click **Analyze Symptoms**.",
                elem_id="results"
            )

    gr.Markdown("""
---
**Responsible use:** This is an educational prototype, not a medical diagnostic service.
Model scores are not calibrated medical probabilities. Avoid entering private medical information.
""")

    analyze_btn.click(fn=predict_health, inputs=symptom_picker, outputs=results)
    clear_btn.click(
        fn=lambda: ([], "## Your results will appear here\n\nSelect symptoms and click **Analyze Symptoms**."),
        inputs=None,
        outputs=[symptom_picker, results]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port)
