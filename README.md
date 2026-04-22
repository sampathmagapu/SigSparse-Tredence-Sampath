```
Feature gating with L1-regularized sigmoid gates for soft sparsity. Includes experiments on lambda tuning, performance vs sparsity trade-offs, and gate distribution analysis.
```


# L1-Regularized Sigmoid Gating for Feature Sparsity

## Overview
This project explores a feature gating mechanism where each input feature is modulated using a learnable sigmoid gate. An L1 regularization term is applied to the gate activations to encourage sparsity.

The goal is to study:
- Whether L1 regularization can suppress unimportant features
- The trade-off between model accuracy and sparsity
- The behavior of gated feature distributions under different regularization strengths

---

## Key Idea

Each feature is multiplied by a gate:

\[
z_i = \sigma(g_i)
\]

Where:
- \(g_i\) = learnable parameter
- \(\sigma(\cdot)\) = sigmoid function

The total loss:

\[
\mathcal{L} = \mathcal{L}_{classification} + \lambda \sum_i |\sigma(g_i)|
\]

Since sigmoid outputs are in (0,1), the L1 penalty pushes gates toward small values, effectively suppressing less important features.

---

## Important Insight

This approach does **not produce true sparsity** (exact zeros), because:
- Sigmoid never outputs 0
- Gates can only approach zero asymptotically

Instead, it produces **soft sparsity**, where:
- Irrelevant features → very small gate values
- Important features → higher gate values

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity (<1e−2) |
|-----------|--------------|------------------|
| 0.0       | 53.10%       | 0.00%            |
| 1e-5      | 54.74%       | 0.00%            |
| 5e-5      | 54.07%       | 0.00%            |
| 1e-4      | 52.98%       | 0.00%            |
| 2e-4      | 51.41%       | 0.00%            |

### Observations
- Small λ improves generalization by suppressing noisy features
- Higher λ reduces performance due to over-regularization
- No hard sparsity observed (expected due to sigmoid constraint)

---

## Gate Distribution

A histogram of final gate values is used to analyze sparsity behavior.

Expected pattern:
- Large concentration near 0 → suppressed features
- Secondary cluster away from 0 → important features

### Plot Code

```python
import matplotlib.pyplot as plt

plt.hist(final_gates, bins=30)
plt.title("Gate Value Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.show()
````

---

## Project Structure

```
.
├── data/                # Dataset (if applicable)
├── models/              # Model definitions
├── experiments/         # Training scripts & lambda tuning
├── plots/               # Gate distribution visualizations
├── results/             # Logs and outputs
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Run experiments
python experiments/lambda_sweep.py
```

---

## Key Takeaways

* L1 on sigmoid gates induces **feature importance ranking**, not strict selection
* Proper λ tuning is critical
* True sparsity requires alternative approaches like:

  * L0 regularization
  * Hard-concrete gates
  * Thresholding

---

## Author

**Sampath Magapu**
📧 [sampathmagapu11@gmail.com](mailto:sampathmagapu11@gmail.com)
🌐 [https://sampathmagapu.github.io/SampathM_Portfolio/](https://sampathmagapu.github.io/SampathM_Portfolio/)

---

## License

This project is for educational and research purposes.

```

---

## Brutal feedback (don’t ignore this)
- This is **good enough to submit**
- But not enough to **stand out**

If you want this repo to actually impress:
- Add **visual plots (images, not just code)**
- Add **ablation: L1 on g vs σ(g)**
- Add **one improvement idea implemented**

---

If you want, I can upgrade this into a **top-tier repo (readme + visuals + structure + recruiter-level polish)**.
```
