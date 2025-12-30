# visualisation.py

import os
import json
import numpy as np
import joblib

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix  # [web:20][web:25]

DEFAULT_COLORS = {
    "bg": "#0f172a",
    "accent": "#6366f1",
    "success": "#10b981",
    "danger": "#f43f5e",
    "warning": "#f59e0b",
    "grid": "#1e293b",
}


def _apply_style(fig, title, subtitle=""):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DEFAULT_COLORS["bg"],
        plot_bgcolor=DEFAULT_COLORS["bg"],
        title={
            "text": (
                f"<span style='font-size:22px; font-weight:800;'>{title}</span>"
                f"<br><span style='font-size:14px; color:#94a3b8;'>{subtitle}</span>"
            ),
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
        },
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color="#f8fafc",
            size=12,
        ),
        margin=dict(l=70, r=70, t=130, b=70),
    )


def _load_best_model(project, model_dir, best_dict):
    best_name = best_dict.get("name")
    if not best_name:
        raise RuntimeError("No best model name in best dict")

    deployed_dir = os.path.join(model_dir, "deployed_model")
    if os.path.isdir(deployed_dir):
        # pick first pkl/joblib in deployed_dir
        for fn in os.listdir(deployed_dir):
            if fn.endswith(".pkl") or fn.endswith(".joblib"):
                path = os.path.join(deployed_dir, fn)
                return joblib.load(path), best_name

    candidates = [
        os.path.join(model_dir, f"{best_name}_model.pkl"),
        os.path.join(model_dir, f"{best_name}_model.joblib"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return joblib.load(c), best_name

    raise RuntimeError(f"Could not find saved model for {best_name}")


def _get_feature_names(model, X_test):

    n_feats = X_test.shape[1]
    return [f"feat_{i}" for i in range(n_feats)]


def generate_visualizations(project, model_dir, X_test, y_test, results, best):
    """
    project   : 'diabetes' or 'heart'
    model_dir : path like 'models/diabetes'
    X_test, y_test : numpy arrays
    results   : list of result dicts from trainandevaluate.py
    best      : 'best' dict from last_run_summary.json
    """
    output_path = os.path.join(model_dir, "visualizations")
    os.makedirs(output_path, exist_ok=True)

    best_model, best_model_name = _load_best_model(project, model_dir, best)
    feature_names = _get_feature_names(best_model, X_test)

    # Build simple model_results dict from `results`
    model_results = {}
    for r in results:
        if "error" in r:
            continue
        name = r["name"]
        model_results[name] = {
            "metrics": {
                "recall": float(r.get("recall", 0.0)),
                "f1_score": float(r.get("f1_score", 0.0)),
                "accuracy": float(r.get("accuracy", 0.0)),
                "inference_latency": float(r.get("latency", 0.0)),
            }
        }

    # 1) Pareto 3D: recall vs f1 vs latency
    models = list(model_results.keys())
    recall_vals = [model_results[m]["metrics"]["recall"] for m in models]
    f1_vals = [model_results[m]["metrics"]["f1_score"] for m in models]
    lat_vals = [model_results[m]["metrics"]["inference_latency"] for m in models]

    fig_pareto = go.Figure(
        data=[
            go.Scatter3d(
                x=recall_vals,
                y=f1_vals,
                z=lat_vals,
                mode="markers+text",
                text=models,
                marker=dict(
                    size=12,
                    color=lat_vals,
                    colorscale="Magma",
                    reversescale=True,
                    opacity=0.9,
                    colorbar=dict(title="Latency (s)"),
                ),
            )
        ]
    )
    fig_pareto.update_layout(
        scene=dict(
            xaxis_title="Recall",
            yaxis_title="F1 Score",
            zaxis_title="Latency (s)",
        )
    )
    _apply_style(
        fig_pareto,
        f"{project.upper()}: Pareto 3D",
        "Recall vs F1 vs Latency",
    )
    fig_pareto.write_html(os.path.join(output_path, "01_pareto_3d.html"))  # [web:6][web:16]

    # 2) Latency vs Recall tradeoff (2D)
    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(
        go.Scatter(
            x=lat_vals,
            y=recall_vals,
            mode="markers+text",
            text=models,
            marker=dict(
                size=16,
                color=recall_vals,
                colorscale="Viridis",
                showscale=True,
            ),
        )
    )
    fig_tradeoff.update_xaxes(title_text="Latency (s)")
    fig_tradeoff.update_yaxes(title_text="Recall")
    _apply_style(
        fig_tradeoff,
        f"{project.upper()}: Latency vs Recall",
        "Speed vs sensitivity tradeoff",
    )
    fig_tradeoff.write_html(
        os.path.join(output_path, "02_latency_recall_tradeoff.html")
    )  # [web:6][web:16]

    # 3) Confusion matrix of best model
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)  # [web:20][web:25]

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale=[[0, "#1e293b"], [1, DEFAULT_COLORS["accent"]],
                        ],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            showscale=False,
        )
    )
    _apply_style(
        fig_cm,
        f"{project.upper()}: Confusion Matrix ({best_model_name})",
        "True vs predicted labels",
    )
    fig_cm.write_html(os.path.join(output_path, "03_confusion_matrix.html"))  # [web:6][web:16]

    # 4) Feature importance (if supported)
    actual_model = getattr(best_model, "model", best_model)
    if hasattr(actual_model, "feature_importances_"):
        imp = np.array(actual_model.feature_importances_)
        idx = np.argsort(imp)

        fig_feat = go.Figure(
            go.Bar(
                x=imp[idx],
                y=[feature_names[j] for j in idx],
                orientation="h",
                marker_color=DEFAULT_COLORS["accent"],
            )
        )
        _apply_style(
            fig_feat,
            f"{project.upper()}: Feature Importance",
            "Top features driving risk",
        )
        fig_feat.write_html(
            os.path.join(output_path, "04_feature_importance.html")
        )  # [web:6][web:16]
    else:
        print("[VIS] Best model has no feature_importances_, skipping feature importance plot")
