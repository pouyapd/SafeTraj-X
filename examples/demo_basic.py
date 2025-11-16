"""
Minimal example for running SafeTrajEvaluator from the command line.
"""
import os
import sys

# Add project root to sys.path so that "import safetraj" works
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from safetraj import SafeTrajEvaluator, SafeTrajConfig, plot_trajectory, plot_feature_importance
import matplotlib.pyplot as plt


def main() -> None:
    config = SafeTrajConfig(seed=42)
    evaluator = SafeTrajEvaluator(config=config)

    x = [0.5, 1.2, 0.3]  # [orientation, v_lin, v_rot]
    res = evaluator.evaluate(x, return_traj=True)

    print("Input:", res["input"])
    print("Risk label:", res["risk_label"])
    print("Risk score:", res["risk_score"])
    print("Mahalanobis:", res["mahalanobis_score"])
    print("IsolationForest:", res["isolation_forest_score"])
    print("Feature importance:", res["feature_importance"])

    fig_traj = plot_trajectory(res["trajectory"])
    fig_imp = plot_feature_importance(res["feature_importance"])

    plt.show()


if __name__ == "__main__":
    main()
