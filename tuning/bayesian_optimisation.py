# Member 1: tunes the best Phase 4 model with Bayesian optimisation (Optuna TPE) using grouped CV on train
import os
import json
import time
import argparse
import pandas as pd
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from config import FEATURES_DIR, TUNING_DIR, N_FOLDS, SEED
from features.feature_builder import FEATURE_COLS
from models.model_definitions import best_model_name
from models.train_models import labelled_only

# Phase 4 settings, run as trial 0 so the untuned model is scored the same way
UNTUNED_PARAMS = {'learning_rate': 0.1, 'max_iter': 100, 'max_depth': 8,
                  'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'l2_regularization': 0.0}

# each range contains the untuned value; max_iter goes above 100 because the Phase 4
# validation loss was still falling at iteration 100 (results/models/history_Hist_Gradient_Boosting.csv)
SEARCH_SPACE = {
    'learning_rate': 'float 0.02 - 0.3, log scale',
    'max_iter': 'int 100 - 300, step 50',
    'max_depth': 'int 4 - 12',
    'max_leaf_nodes': 'int 15 - 63',
    'min_samples_leaf': 'int 20 - 200',
    'l2_regularization': 'float 0 - 1',
}


def suggest_params(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 200),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
    }


def make_model(params):
    return HistGradientBoostingClassifier(**params, early_stopping=True, random_state=SEED)


def cv_macro_f1(params, df):
    # grouped CV: each trip stays in one fold; scored on labelled rows only, threshold 0.5
    scores = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(df, groups=df['trip_id']):
        fit_df, va_df = labelled_only(df.iloc[tr]), labelled_only(df.iloc[va])
        model = make_model(params).fit(fit_df[FEATURE_COLS], fit_df['target'])
        scores.append(f1_score(va_df['target'], model.predict(va_df[FEATURE_COLS]), average='macro'))
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    os.makedirs(TUNING_DIR, exist_ok=True)

    model_name = best_model_name()  # highest grouped-CV Macro F1 in results/models/model_comparison.csv
    if model_name != 'Hist_Gradient_Boosting':
        raise SystemExit(f"Best Phase 4 model is {model_name}; this search space is for HistGradientBoosting")

    df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    print(f"Tuning {model_name}: {args.trials} trials, grouped {N_FOLDS}-fold CV Macro F1 on the 80 train trips")

    def objective(trial):
        score = cv_macro_f1(suggest_params(trial), df)
        print(f"  trial {trial.number}: CV macro F1={score:.4f}")
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(UNTUNED_PARAMS)
    start = time.perf_counter()
    study.optimize(objective, n_trials=args.trials)
    total = time.perf_counter() - start

    study.trials_dataframe().to_csv(TUNING_DIR / "optuna_trials.csv", index=False)
    summary = {
        'model': model_name,
        'sampler': 'Optuna TPESampler (Bayesian), seed 42',
        'objective': f'mean grouped {N_FOLDS}-fold CV Macro F1 on the 80 train trips',
        'search_space': SEARCH_SPACE,
        'n_trials': args.trials,
        'untuned_cv_macro_f1': round(study.trials[0].value, 4),
        'best_params': study.best_params,
        'best_cv_macro_f1': round(study.best_value, 4),
        'total_time_s': round(total, 1),
        'time_per_trial_s': round(total / args.trials, 1),
    }
    with open(TUNING_DIR / "best_params.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nUntuned CV macro F1: {summary['untuned_cv_macro_f1']} | best: {summary['best_cv_macro_f1']}")
    print(f"Best params: {study.best_params}")
    print(f"Cost: {total / 60:.1f} min total, {total / args.trials:.1f} s per trial")


if __name__ == "__main__":
    main()
