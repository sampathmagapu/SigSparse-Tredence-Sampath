# SigSparse – L1-Regularized Sigmoid Gating for Soft Sparsity

**Author:** Sampath Magapu  
**Email:** sampathmagapu11@gmail.com  
**Portfolio:** https://sampathmagapu.github.io/SampathM_Portfolio/

---

## 📌 Overview

This project implements a **feature gating mechanism** where each input feature is modulated by a learnable sigmoid gate. An L1 regularization term is applied to the gate activations to encourage sparsity.

The core idea:

```
z_i = σ(g_i)          # gate value for feature i
Loss = ClassificationLoss + λ · Σ σ(g_i)
```

Because sigmoid outputs lie in (0,1), the L1 penalty pushes gates toward small values, suppressing less important features. This results in **soft sparsity** – feature importance ranking rather than exact zeroing.

---

## 🆕 What’s New in the `additional-feature` Branch

This branch adds two production‑ready components to the original research project:

1. **Interactive Dashboard (Streamlit)**  
   - Visualise experiment results (accuracy, sparsity, gate statistics)  
   - Dynamically load gate distributions for different λ values  
   - Compare pre‑generated plots  
   - **Live image classification** using the trained model (no retraining)

2. **Model Serving API (FastAPI)**  
   - REST endpoint `/predict` – accepts an image and returns the predicted class  
   - Lightweight, CPU‑only inference  
   - Easily integrable into other applications

> ✅ **No retraining** – the saved model (`model.pth`) is loaded once and used only for inference.

---

## 🧠 Core Research Findings

| Lambda (λ) | Test Accuracy | Sparsity (<1e‑2) | Gate Mean |
|------------|--------------|------------------|-----------|
| 0.0        | 53.10%       | 0.00%            | 0.4880    |
| 1e-5       | 54.74%       | 0.00%            | 0.2468    |
| 5e-5       | 54.07%       | 0.00%            | 0.1084    |
| 1e-4       | 52.98%       | 0.00%            | 0.0806    |
| 2e-4       | 51.41%       | 0.00%            | 0.0653    |

**Observations:**
- Small λ slightly improves generalisation (suppresses noise)  
- Higher λ reduces accuracy due to over‑regularisation  
- No **hard** sparsity (exact zeros) – sigmoid outputs never reach 0  
- This is **soft sparsity**: gates approach ~0.05 but remain positive

---

## 📁 Project Structure (additional-feature branch)

```
SigSparse-Tredence-Sampath/
├── app.py                     # Streamlit dashboard
├── api.py                     # FastAPI inference server
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── save_data/                 # Trained model & experiment artifacts
│   ├── model.pth              # Trained weights (no retraining)
│   ├── results.json           # All λ‑sweep results
│   ├── gates/                 # Raw gate arrays per λ
│   └── plots/                 # Pre‑generated histograms
├── notebook/                  # Original Jupyter notebook
│   └── SelfPruningModelSampath.ipynb
└── report/                    # Project documentation
    └── SigSparse_Sampath_TA_Report.docx
```

---

## 🚀 How to Run the Additional Features

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Then open the displayed URL (usually `http://localhost:8501`).

**What you can do:**
- Select different λ values from the sidebar
- View accuracy, sparsity, and gate statistics
- See the gate distribution histogram (dynamic)
- Compare pre‑generated plots
- Upload your own image and get a live prediction

### 3. Start the FastAPI Server
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
**Test the API:**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@test.jpg"
```
Or visit `http://localhost:8000/docs` for interactive Swagger documentation.

---

## 🧪 Running the Original Experiments (Training)

If you want to reproduce the λ‑sweep or retrain the model, use the original notebook:

```bash
jupyter notebook notebook/SelfPruningModelSampath.ipynb
```

**Note:** The `additional-feature` branch **does not retrain** – it only uses the already saved `model.pth`.

---

## 📊 Visual Examples

| Streamlit Dashboard | FastAPI Swagger |
|--------------------|----------------|
| ![Streamlit UI](docs/streamlit_demo.png) | ![FastAPI Docs](docs/fastapi_demo.png) |

*(Replace with actual screenshots if available)*

---

## 🔬 Technical Deep Dive

### Why Soft Sparsity?
- Sigmoid function: `σ(x) = 1/(1+e^(-x))` → always in (0,1)  
- L1 penalty minimises `Σ σ(g_i)`, pushing gates toward 0  
- But `σ(g_i) = 0` would require `g_i → -∞` – impossible in practice  
- Result: **feature suppression**, not elimination  

### True Sparsity Alternatives
- Hard‑concrete gates (differentiable relaxation of L0)  
- Post‑training thresholding + fine‑tuning  
- Straight‑through estimator (STE)

---

## 📄 License

This project is for educational and research purposes only.

---

## 🙏 Acknowledgements

- CIFAR‑10 dataset  
- PyTorch & Streamlit communities  
- Tredence AI Engineering Internship case study

```

---
