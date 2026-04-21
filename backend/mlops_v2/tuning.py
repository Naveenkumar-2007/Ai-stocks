from __future__ import annotations

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


def sharpe_ratio_from_returns(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    std = np.std(returns)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std)


def tune_xgb_for_sharpe(X, y, forward_returns, n_trials: int = 50):
    """Weekly tuning target: maximize validation Sharpe ratio."""

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
        }

        model = XGBClassifier(**params)
        cv = TimeSeriesSplit(n_splits=3)
        sharpe_scores = []

        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            val_returns = forward_returns.iloc[val_idx].values

            model.fit(X_train, y_train, verbose=False)
            probs = model.predict_proba(X_val)[:, 1]
            signal = np.where(probs >= 0.5, 1.0, -1.0)
            strategy_returns = signal * val_returns
            sharpe_scores.append(sharpe_ratio_from_returns(strategy_returns))

        return float(np.mean(sharpe_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)
