from safetraj import SafeTrajEvaluator, SafeTrajConfig


def test_evaluate_runs():
    config = SafeTrajConfig(seed=0)
    ev = SafeTrajEvaluator(config=config)

    res = ev.evaluate([0.0, 0.5, 0.0])

    assert "risk_score" in res
    assert "risk_label" in res
    assert isinstance(res["risk_score"], float)
