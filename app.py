# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import math
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. BASE PATH – updated to match your Git repo structure
# ----------------------------------------------------------------------
BASE_PATH = r"D:\AI_Field\Deep_Learning_projects\SigSparse\SigSparse-Tredence-Sampath\save_data"

# ----------------- Model definition (same as before) -----------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

    def forward(self, x):
        gates = torch.sigmoid(2 * self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

class SelfPruningNet(nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        self.fc1 = PrunableLinear(3072, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------- Helper functions -----------------
def load_model():
    model = SelfPruningNet()
    model_path = os.path.join(BASE_PATH, "model.pth")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_results():
    json_path = os.path.join(BASE_PATH, "results.json")
    if not os.path.exists(json_path):
        st.error(f"Results file not found at {json_path}")
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_gates(lambda_value):
    lam_str = str(lambda_value)   # "0.0", "1e-05", "5e-05", "0.0001", "0.0002"
    filename = f"gates_lambda_{lam_str}.npy"
    filepath = os.path.join(BASE_PATH, "gates", filename)
    if not os.path.exists(filepath):
        st.error(f"Gate file not found: {filepath}")
        return None
    return np.load(filepath)

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img_array = (img_array - mean) / std
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="SigSparse - Self-Pruning NN", layout="wide")
st.title("🧠 SigSparse: Self-Pruning Neural Network")
st.markdown("### Soft Sparsity via Learnable Gating – Lambda vs Accuracy Trade-off")

results_df = load_results()
model = load_model()
if results_df is None or model is None:
    st.stop()

# Sidebar
st.sidebar.header("Experiment Selector")
lambda_options = results_df['lambda'].tolist()
lambda_display = [str(l) for l in lambda_options]
selected_idx = st.sidebar.selectbox("Choose λ", range(len(lambda_options)), format_func=lambda i: lambda_display[i])
selected_lambda = lambda_options[selected_idx]
gates = load_gates(selected_lambda)

col1, col2 = st.columns(2)

# Section A: Metrics
with col1:
    st.subheader(f"📊 Metrics for λ = {selected_lambda}")
    row = results_df[results_df['lambda'] == selected_lambda].iloc[0]
    acc = row['accuracy']
    sp = row['sparsity']
    g_mean = row['gate_mean']
    g_min = row['gate_min']
    g_max = row['gate_max']
    
    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric("Accuracy (%)", f"{acc:.2f}")
    col_metric2.metric("Sparsity (%)", f"{sp:.2f}")
    col_metric1.metric("Gate Mean", f"{g_mean:.4f}")
    col_metric2.metric("Gate Min / Max", f"{g_min:.4f} / {g_max:.4f}")
    st.info("ℹ️ **Soft Sparsity** – Gates are sigmoid outputs in (0,1). They approach small values (~0.05) but never reach 0. This suppresses weak connections without hard removal.")

# Section B: Table (manual HTML, no pyarrow)
with col2:
    st.subheader("📋 All Experiment Results")
    html_table = results_df.to_html(index=False, border=0, classes='dataframe')
    st.markdown(f"""
    <style>
    .dataframe {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    .dataframe th, .dataframe td {{
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }}
    .dataframe th {{
        background-color: #f2f2f2;
    }}
    </style>
    {html_table}
    """, unsafe_allow_html=True)

# Section C: Dynamic histogram
st.subheader(f"📈 Gate Distribution Histogram (λ = {selected_lambda})")
if gates is not None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(gates, bins=100, color='royalblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.01, color='red', linestyle='--', label='Threshold (0.01)')
    ax.set_xlim(0, 0.2)
    ax.set_xlabel("Gate Value (sigmoid output)")
    ax.set_ylabel("Number of Weights")
    ax.set_title(f"Gate Distribution for λ = {selected_lambda}")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Gate data not available for this lambda.")

# Section D: Pre-generated plot
st.subheader(f"🖼️ Pre-generated Gate Distribution (λ = {selected_lambda})")
lam_str_plot = str(selected_lambda)
plot_path = os.path.join(BASE_PATH, "plots", f"gate_dist_{lam_str_plot}.png")
if os.path.exists(plot_path):
    st.image(plot_path, caption=f"Pre-generated plot for λ={selected_lambda}", use_container_width=True)
else:
    st.warning(f"Pre-generated plot not found at {plot_path}")

# Section E: Comparison plot
st.subheader("🔍 Lambda Comparison: Gate Distributions Overlay")
comparison_path = os.path.join(BASE_PATH, "plots", "lambda_comparison.png")
if os.path.exists(comparison_path):
    st.image(comparison_path, caption="Overlay of gate distributions for all λ values", use_container_width=True)
else:
    st.warning("Comparison plot not found. Run the notebook to generate it.")

# Live prediction
st.markdown("---")
st.header("🚀 Live Prediction with Trained Model")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    col_img, col_pred = st.columns(2)
    with col_img:
        st.image(uploaded_file, caption="Uploaded Image", width=200)
    with col_pred:
        input_tensor = preprocess_image(uploaded_file)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            class_idx = predicted.item()
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        pred_class = classes[class_idx]
        st.success(f"**Predicted Class:** {pred_class} (index {class_idx})")
        st.caption("Note: Model is trained on CIFAR-10 (32×32 low-res images). Performance on arbitrary images may vary.")

# Explanation expander
with st.expander("📖 About Soft Sparsity & This Project"):
    st.markdown("""
    **Why soft sparsity?**  
    - Each weight has a learnable `gate_score` transformed via sigmoid → gate ∈ (0,1).  
    - The loss = CrossEntropy + λ × sum(gates).  
    - Larger λ pushes gates toward zero, but sigmoid prevents exact zeros.  
    - This creates **weight suppression** rather than hard pruning.  
    
    **Trade-off observed:**  
    - λ = 0 → high accuracy, no sparsity.  
    - λ increases → gates shrink (mean ~0.05-0.1), accuracy drops slightly due to over-regularization.  
    - True hard sparsity would require alternative gating (e.g., hard‑concrete).  
    
    **Why this matters:**  
    - Demonstrates learnable regularization directly inside the model.  
    - Useful for understanding network redundancy and building efficient models.
    """)

st.markdown("---")
st.caption("SigSparse – Self-Pruning Neural Network | Built with Streamlit")