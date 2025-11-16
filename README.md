# SafeTraj-X

A small, self-contained tool for **trajectory prediction** and **risk estimation** in mobile robots / smart wheelchairs.

Give it a motion command:

- orientation (rad)  
- linear velocity (m/s)  
- rotational velocity (rad/s)  

and SafeTraj-X will:

1. Predict the future trajectory using a simple kinematic model  
2. Estimate how *unusual* this command is using two input-space OOD detectors  
3. Combine them into a **risk score** and a human-readable **risk label**  
4. Visualize the trajectory and a simple feature-importance view

It is not a full planner.  
It is a **safety & explainability layer** that can sit on top of an existing controller and say:  
> “This command looks in-distribution & low-risk”  
> or  
> “This is highly OOD / high-risk – better block or review it.”

---

## Why does this exist?

Most safety experiments in ML for robotics live inside messy Jupyter notebooks.  
You tweak a few cells, plot a figure, and then the code dies there.

SafeTraj-X tries to be something else:

- a **mini-library** you can import in other projects  
- a **repeatable** experiment with fixed seeds  
- a **small dashboard** you can show to supervisors, colleagues or in an interview  
- a compact example of combining:
  - trajectory prediction  
  - OOD detection  
  - simple explainability  
  - a clean Python API

---

## Features

- **Kinematic trajectory predictor**  
  - Straightforward model: \[x(t), y(t), θ(t)\] from \[orientation, v_lin, v_rot\]  
  - Configurable horizon and number of steps

- **Two OOD detectors in input space**
  - Mahalanobis distance (EmpiricalCovariance)  
  - Isolation Forest (scikit-learn)

- **Risk layer on top of OOD**
  - Combines the two scores into one `risk_score`  
  - Maps them to discrete labels:
    - `in-distribution & low-risk`
    - `borderline / uncertain`
    - `high OOD / high-risk`

- **Simple explainability**
  - Heuristic feature-importance for the three input features  
  - Intended as an intuitive visualization, not a full XAI method

- **Interactive dashboard (Streamlit)**
  - Sliders for the three inputs  
  - Live trajectory plot  
  - Live feature-importance plot  
  - Risk summary and raw JSON output

- **Modern project structure**
  - Modular package (`safetraj/`)  
  - `examples/` for quick CLI demos  
  - `dashboard/` for the web UI  
  - `tests/` with a minimal pytest check

---

## Project structure

```text
SafeTraj-X/
├─ safetraj/
│  ├─ __init__.py          # public API (SafeTrajEvaluator, config, plotting)
│  ├─ config.py            # configuration dataclass
│  ├─ data.py              # synthetic training data generator
│  ├─ predictor.py         # kinematic trajectory model
│  ├─ ood.py               # Mahalanobis + Isolation Forest OOD detectors
│  ├─ xai.py               # simple feature-importance heuristic
│  ├─ evaluator.py         # high-level risk evaluator
│  └─ plotting.py          # Matplotlib visualizations
├─ dashboard/
│  └─ app.py               # Streamlit dashboard
├─ examples/
│  └─ demo_basic.py        # command-line demo
├─ tests/
│  └─ test_evaluator.py    # minimal pytest sanity check
├─ requirements.txt
└─ README.md
