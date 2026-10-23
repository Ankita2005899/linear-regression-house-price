"""Metrics + diagnostic plots."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from . import config

def metrics(y_true, y_pred, n_features=None):
    r2 = r2_score(y_true, y_pred)
    out = {"MAE": mean_absolute_error(y_true, y_pred),
           "MSE": mean_squared_error(y_true, y_pred),
           "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
           "R2": r2,
           "MAPE": float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true))))}
    if n_features:
        n = len(y_true)
        out["Adj_R2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return out

def diagnostic_plots(y_true, y_pred, name="model", note=""):
    config.OUT_DIR.mkdir(exist_ok=True)
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(y_true, y_pred, alpha=.5); lim = [min(y_true), max(y_true)]
    ax[0].plot(lim, lim, "r--"); ax[0].set(title="Actual vs Predicted" + note, xlabel="Actual", ylabel="Predicted")
    ax[1].scatter(y_pred, resid, alpha=.5); ax[1].axhline(0, color="r", ls="--")
    ax[1].set(title="Residuals vs Predicted (look for no pattern)" + note, xlabel="Predicted", ylabel="Residual")
    stats.probplot(resid, plot=ax[2]); ax[2].set_title("Q-Q plot (normality of residuals)" + note)
    plt.tight_layout(); plt.savefig(config.OUT_DIR / f"{name}_diagnostics.png", dpi=120); plt.close()

def plot_feature_importance(names, coefs, name="model"):
    config.OUT_DIR.mkdir(exist_ok=True)
    order = np.argsort(np.abs(coefs))
    plt.figure(figsize=(8, 5)); plt.barh(np.array(names)[order], np.array(coefs)[order])
    plt.title("Standardised coefficients (impact on price)"); plt.tight_layout()
    plt.savefig(config.OUT_DIR / f"{name}_coefficients.png", dpi=120); plt.close()

def plot_loss(history):
    config.OUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4)); plt.plot(history); plt.xlabel("Iteration"); plt.ylabel("MSE")
    plt.title("Gradient descent convergence"); plt.tight_layout()
    plt.savefig(config.OUT_DIR / "gradient_descent_loss.png", dpi=120); plt.close()
