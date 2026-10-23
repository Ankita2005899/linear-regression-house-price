"""Train, compare and save the best model on the real Mumbai listings.
Models: OLS Linear, Ridge (L2), Lasso (L1), ElasticNet, Polynomial(2)+Ridge.
The price is modelled on a log scale (house prices are skewed), then converted back to lakh rupees.
Uses a Pipeline (no data leakage), K-fold CV and GridSearchCV for hyper-parameters."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline

from . import config
from .evaluate import metrics, diagnostic_plots, plot_feature_importance, plot_loss
from .from_scratch import LinearRegressionScratch
from .preprocessing import build_preprocessor, load_data

def wrap(pipe):
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

def main():
    config.OUT_DIR.mkdir(exist_ok=True); config.MODEL_PATH.parent.mkdir(exist_ok=True)
    df = load_data()                      # outliers were already removed in src/prepare_data.py
    X, y = df[config.NUMERIC + config.CATEGORICAL], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    P = lambda deg, model: wrap(Pipeline([("prep", build_preprocessor(deg)), ("model", model)]))
    a = "regressor__model__alpha"
    candidates = {
        "linear":     (P(1, LinearRegression()), {}),
        "ridge":      (P(1, Ridge()), {a: [0.1, 1, 10, 100]}),
        "lasso":      (P(1, Lasso(max_iter=20000)), {a: [0.0001, 0.001, 0.01]}),
        "elasticnet": (P(1, ElasticNet(max_iter=20000)), {a: [0.0001, 0.001], "regressor__model__l1_ratio": [0.2, 0.8]}),
        "poly2_ridge":(P(2, Ridge()), {a: [1, 10, 100]}),
    }
    cv = KFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    rows, fitted = [], {}
    for name, (pipe, grid) in candidates.items():
        search = GridSearchCV(pipe, grid or {}, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X_tr, y_tr)
        best = search.best_estimator_; pred = best.predict(X_te)
        m = metrics(y_te, pred, n_features=X_te.shape[1])
        rows.append({"model": name, "cv_RMSE": -search.best_score_, "best_params": search.best_params_, **m})
        fitted[name] = best
        print(f"{name:12s} CV-RMSE={-search.best_score_:,.1f}  test RMSE={m['RMSE']:,.1f} lakh  R2={m['R2']:.4f}  MAPE={m['MAPE']:.1%}")

    # choose the winner by cross-validation on TRAIN data (not by peeking at the test set)
    results = pd.DataFrame(rows).sort_values("cv_RMSE")
    results.to_csv(config.OUT_DIR / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["model"]; best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    joblib.dump(best, config.MODEL_PATH)

    pred = best.predict(X_te)
    diagnostic_plots(np.log1p(y_te), np.log1p(pred), best_name, note=" (log price)")
    inner = fitted["ridge"].regressor_      # coefficients are easiest to read on the plain Ridge model
    names = list(inner.named_steps["prep"].get_feature_names_out()); coefs = inner.named_steps["model"].coef_
    top = np.argsort(-np.abs(coefs))[:20]
    plot_feature_importance([names[i] for i in top], coefs[top], "ridge")
    print("\nLargest Ridge coefficients (standardised, effect on log price):")
    for i in top[:10]: print(f"  {names[i]:40s}{coefs[i]:>10.3f}")
    ratio = np.asarray(y_te) / pred          # actual / predicted -> used for the app's "likely range"
    out = {"best_model": best_name, **metrics(y_te, pred),
           "ratio_p10": float(np.percentile(ratio, 10)), "ratio_p90": float(np.percentile(ratio, 90)),
           "n_train": int(len(X_tr)), "n_test": int(len(X_te))}
    json.dump(out, open(config.OUT_DIR / "best_metrics.json", "w"), indent=2)

    # From-scratch comparison (log price) on the same preprocessed data
    prep = build_preprocessor().fit(X_tr)
    dense = lambda M: M.toarray() if hasattr(M, "toarray") else M
    A_tr, A_te = dense(prep.transform(X_tr)), dense(prep.transform(X_te))
    for method in ("normal", "gd"):
        s = LinearRegressionScratch(method=method).fit(A_tr, np.log1p(y_tr))
        m = metrics(y_te, np.expm1(s.predict(A_te)))
        print(f"scratch-{method:6s} RMSE={m['RMSE']:,.1f} lakh  R2={m['R2']:.4f}")
        if method == "gd": plot_loss(s.loss_history)

if __name__ == "__main__":
    main()
