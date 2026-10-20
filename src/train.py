"""Train, compare and save the best model.
Models: OLS Linear, Ridge (L2), Lasso (L1), ElasticNet, Polynomial(2)+Ridge.
Uses a Pipeline (no data leakage), K-fold CV and GridSearchCV for hyper-parameters."""
import json
import joblib
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline

from . import config
from .evaluate import metrics, diagnostic_plots, plot_feature_importance, plot_loss
from .from_scratch import LinearRegressionScratch
from .preprocessing import build_preprocessor, load_data, remove_outliers_iqr

def get_feature_names(pipe):
    return list(pipe.named_steps["prep"].get_feature_names_out())

def main():
    config.OUT_DIR.mkdir(exist_ok=True); config.MODEL_PATH.parent.mkdir(exist_ok=True)
    df = remove_outliers_iqr(load_data(), config.NUMERIC + [config.TARGET])
    X, y = df[config.NUMERIC + config.CATEGORICAL], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    candidates = {
        "linear":     (Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())]), {}),
        "ridge":      (Pipeline([("prep", build_preprocessor()), ("model", Ridge())]), {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
        "lasso":      (Pipeline([("prep", build_preprocessor()), ("model", Lasso(max_iter=20000))]), {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
        "elasticnet": (Pipeline([("prep", build_preprocessor()), ("model", ElasticNet(max_iter=20000))]),
                       {"model__alpha": [0.01, 0.1, 1, 10], "model__l1_ratio": [0.2, 0.5, 0.8]}),
        "poly2_ridge":(Pipeline([("prep", build_preprocessor(2)), ("model", Ridge())]), {"model__alpha": [1, 10, 100]}),
    }
    cv = KFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    rows, fitted = [], {}
    for name, (pipe, grid) in candidates.items():
        search = GridSearchCV(pipe, grid or {}, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X_tr, y_tr)
        best = search.best_estimator_; pred = best.predict(X_te)
        m = metrics(y_te, pred, n_features=X_te.shape[1])
        rows.append({"model": name, "cv_RMSE": -search.best_score_, "best_params": search.best_params_, **m})
        fitted[name] = best
        print(f"{name:12s} CV-RMSE={-search.best_score_:,.0f}  test RMSE={m['RMSE']:,.0f}  R2={m['R2']:.4f}")

    # choose the winner by cross-validation on TRAIN data (not by peeking at the test set)
    results = pd.DataFrame(rows).sort_values("cv_RMSE")
    results.to_csv(config.OUT_DIR / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["model"]; best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    joblib.dump(best, config.MODEL_PATH)

    pred = best.predict(X_te)
    diagnostic_plots(y_te, pred, best_name)
    if best_name in ("linear", "ridge", "lasso", "elasticnet"):
        names = get_feature_names(best); coefs = best.named_steps["model"].coef_
        plot_feature_importance(names, coefs, best_name)
        print("\nCoefficients (standardised):")
        for n, c in sorted(zip(names, coefs), key=lambda t: -abs(t[1])): print(f"  {n:28s}{c:>12,.1f}")
    json.dump({"best_model": best_name, **metrics(y_te, pred)}, open(config.OUT_DIR / "best_metrics.json", "w"), indent=2)

    # From-scratch comparison on the same preprocessed data
    prep = build_preprocessor().fit(X_tr)
    A_tr, A_te = prep.transform(X_tr), prep.transform(X_te)
    for method in ("normal", "gd"):
        s = LinearRegressionScratch(method=method).fit(A_tr, y_tr)
        m = metrics(y_te, s.predict(A_te))
        print(f"scratch-{method:6s} RMSE={m['RMSE']:,.0f}  R2={m['R2']:.4f}")
        if method == "gd": plot_loss(s.loss_history)

if __name__ == "__main__":
    main()
