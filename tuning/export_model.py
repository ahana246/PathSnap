# Member 1: compares tuned vs untuned on the same data and exports the better one as the final model
import os
import json
import time
import joblib
import pandas as pd

from config import STUDENT_TEST_DIR, FEATURES_DIR, TUNING_DIR, ARTIFACTS_DIR, SEED
from features.feature_builder import FEATURE_COLS
from models.evaluator import evaluate, load_arcs_by_trip
from models.train_models import labelled_only
from tuning.bayesian_optimisation import make_model, cv_macro_f1, UNTUNED_PARAMS


def assess(name, params, train_df, test_df, arcs_by_trip):
    # same CV folds, same training rows and same student_test rows for both models
    cv_f1 = cv_macro_f1(params, train_df)

    fit_df = labelled_only(train_df)
    model = make_model(params).fit(fit_df[FEATURE_COLS], fit_df['target'])

    start = time.perf_counter()
    probs = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
    infer_time = time.perf_counter() - start
    test, _ = evaluate(test_df, probs, arcs_by_trip)

    return {'model': name, 'cv_macro_f1': round(cv_f1, 4),
            'test_top1_accuracy': test['top1_accuracy'], 'test_macro_f1': test['macro_f1'],
            'test_log_loss': test['log_loss'],
            'inference_ms_per_obs': round(infer_time / test['observations'] * 1000, 4)}, model


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(TUNING_DIR / "best_params.json") as f:
        tuning = json.load(f)

    train_df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_df = pd.read_parquet(FEATURES_DIR / "student_test_features.parquet")
    arcs_by_trip = load_arcs_by_trip(STUDENT_TEST_DIR, test_df['trip_id'].unique())

    untuned_row, untuned_model = assess('untuned', UNTUNED_PARAMS, train_df, test_df, arcs_by_trip)
    tuned_row, tuned_model = assess('tuned', tuning['best_params'], train_df, test_df, arcs_by_trip)
    table = pd.DataFrame([untuned_row, tuned_row])
    table.to_csv(TUNING_DIR / "tuned_vs_untuned.csv", index=False)

    # keep the tuned model only if it beats the untuned one on grouped-CV Macro F1;
    # student_test is reported, not used to choose
    use_tuned = tuned_row['cv_macro_f1'] > untuned_row['cv_macro_f1']
    final_model = tuned_model if use_tuned else untuned_model
    joblib.dump(final_model, ARTIFACTS_DIR / "final_model.joblib")
    with open(ARTIFACTS_DIR / "model_config.json", 'w') as f:
        json.dump({'model': tuning['model'], 'tuned': bool(use_tuned),
                   'params': tuning['best_params'] if use_tuned else UNTUNED_PARAMS,
                   'features': FEATURE_COLS, 'seed': SEED}, f, indent=2)

    print(table.to_string(index=False))
    print(f"\nExported the {'tuned' if use_tuned else 'untuned'} model to {ARTIFACTS_DIR / 'final_model.joblib'}")


if __name__ == "__main__":
    main()
