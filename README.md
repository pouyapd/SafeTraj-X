# SafeTraj-X

SafeTraj-X is a lightweight and modular framework for **trajectory prediction**, **out-of-distribution (OOD) detection**, and **risk estimation** in mobile robots and smart wheelchairs.

Modern robotic systems must decide whether a motion command is **safe**, **unusual**, or **potentially dangerous** before executing it.  
This decision layer is essential for safe navigation, yet it is often hidden inside large codebases or scattered across notebooks.

**SafeTraj-X was designed to make this safety layer simple, explicit, and reproducible.**

It converts a basic motion command:


into:

- a predicted kinematic trajectory  
- Mahalanobis and Isolation Forest OOD scores  
- a combined risk score  
- a human-readable risk label  
- simple feature-importance  
- optional real-time visualization via Streamlit  

The framework is intentionally compact, making it suitable for **research**, **robot safety analysis**, **explainability studies**, and **interactive demos** for supervisors or interviewers.

---

## 🔍 Why This Project?

Robots and autonomous mobility devices (e.g., smart wheelchairs) operate in unpredictable environments.  
A small change in orientation or velocity might push the system into behaviors it was *not trained for*.

SafeTraj-X provides a clean pipeline to explore these effects:

1. How does a command translate into motion?
2. Is this command “in-distribution” or anomalous?
3. How risky is the predicted behavior?
4. Which input contributed most to the risk?

The goal is not high-fidelity simulation.  
The goal is **clarity, modularity, and safety-awareness** — a foundation that can be extended for advanced research or prototyping.

---

## ✨ Core Features

### **1. Trajectory Prediction**
Lightweight kinematic model → predicts `[x(t), y(t), θ(t)]`.

### **2. OOD Detection**
- Mahalanobis distance using empirical covariance  
- Isolation Forest anomaly score (scikit-learn)

### **3. Risk Estimation**
Both OOD outputs are normalized to `[0, 1]` and merged into:

- `risk_score`  
- `risk_label`:
  - low-risk  
  - borderline  
  - high-risk  

### **4. Explainability**
Simple feature-importance showing which input contributed most.

### **5. Interactive Dashboard**
Streamlit app with sliders → real-time trajectory + risk response.

---

## 📦 Installation

Clone the project:

```bash
git clone https://github.com/pouyapd/SafeTraj-X.git
cd SafeTraj-X
pip install -r requirements.txt
pip install -e .
Install dependencies:

pip install -r requirements.txt
pip install -e .


Launch the dashboard:

streamlit run dashboard/app.py

🚀 Quick Example (Python)
from safetraj import SafeTrajEvaluator

# Motion command: [orientation, linear_velocity, angular_velocity]
cmd = [0.5, 1.2, -0.3]

evaluator = SafeTrajEvaluator()
result = evaluator.evaluate(cmd, return_traj=True)

print("Risk label:", result["risk_label"])
print("Risk score:", result["risk_score"])
print("Trajectory shape:", result["trajectory"].shape)
print("Feature importance:", result["feature_importance"])

🧠 Understanding the Risk Score

SafeTraj-X uses a simple but effective scoring method:

Mahalanobis Distance

Measures how far the input is from the training distribution.
High distance → more unusual → riskier.

Isolation Forest

Detects anomaly patterns using ensemble decision trees.

Normalization + Combination

Both outputs are normalized to [0, 1] and combined equally (0.5 / 0.5).

Default Thresholds

< 0.33 → low-risk

< 0.66 → borderline

≥ 0.66 → high-risk

These values can be easily altered in config.py.

📁 Project Structure
SafeTraj-X/
│
├── safetraj/
│   ├── __init__.py
│   ├── config.py            # Configuration dataclass
│   ├── data.py              # Training data generation
│   ├── predictor.py         # Kinematic trajectory predictor
│   ├── ood.py               # Mahalanobis + Isolation Forest detectors
│   ├── xai.py               # Feature importance utilities
│   ├── evaluator.py         # Main high-level API
│   └── plotting.py          # Matplotlib plotting helpers
│
├── dashboard/
│   └── app.py               # Streamlit interactive dashboard
│
├── examples/
│   └── demo_basic.py        # CLI example
│
├── tests/
│   └── test_evaluator.py    # Basic sanity test
│
└── README.md

📷 (Optional) Dashboard Preview

You can include a screenshot like:

![Dashboard Example](dashboard_preview.png)

🧩 Extending the Project

SafeTraj-X is intentionally simple so you can extend it:

replace the predictor with a learned neural model

add new OOD detectors

integrate real robot sensor data

adjust risk scoring logic

plug into ROS / navigation stacks

It’s a clean template for advanced safety-aware research.

📜 License

MIT License
