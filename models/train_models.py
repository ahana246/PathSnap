# Trains all five models with grouped CV on train, evaluates on student_test and saves results
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import (TRAIN_DIR, STUDENT_TEST_DIR, FEATURES_DIR, MODELS_DIR, MODEL_PLOTS_DIR,
                    PREDICTIONS_DIR, N_FOLDS, SEED)
from features.feature_builder import FEATURE_COLS
from models.evaluator import evaluate, load_arcs_by_trip
from models.model_definitions import get_model_catalog


def labelled_only(df):
    # points not covered by the reference route (true_arc = -1) have no reliable label
    return df[df['true_arc'] >= 0]


def cross_validate(name, model, train_df, arcs_by_trip):
    # GroupKFold keeps every trip entirely in one fold, so no trip leaks between train and validation
    rows = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr, va) in enumerate(gkf.split(train_df, groups=train_df['trip_id']), 1):
        fit_df = labelled_only(train_df.iloc[tr])
        m = clone(model).fit(fit_df[FEATURE_COLS], fit_df['target'])
        probs = m.predict_proba(train_df.iloc[va][FEATURE_COLS])[:, 1]
        metrics, _ = evaluate(train_df.iloc[va], probs, arcs_by_trip)
        rows.append({'model': name, 'fold': fold, **metrics})
        print(f"  fold {fold}: macro F1={metrics['macro_f1']:.4f}, top-1={metrics['top1_accuracy']:.4f}")
    return rows


def save_confusion_matrix(name, df, probs):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(df['target'], (probs >= 0.5).astype(int),
                                            display_labels=['not true arc', 'true arc'], ax=ax, colorbar=False)
    ax.set_title(f"{name} (student_test)")
    plt.tight_layout()
    plt.savefig(MODEL_PLOTS_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close()


def save_hgb_history(model):
    # HistGradientBoosting stores the loss per boosting iteration when early stopping is on
    train_loss, val_loss = -model.train_score_, -model.validation_score_
    pd.DataFrame({'iteration': range(len(train_loss)), 'train_loss': train_loss, 'val_loss': val_loss}) \
        .to_csv(MODELS_DIR / "history_Hist_Gradient_Boosting.csv", index=False)
    plot_history("Hist_Gradient_Boosting", train_loss, val_loss, "Boosting iteration")


def mlp_loss_history(train_df, epochs=50):
    # MLP trained epoch by epoch on 80% of training trips, validated on the other 20% of trips
    train_df = labelled_only(train_df).reset_index(drop=True)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                  .split(train_df, groups=train_df['trip_id']))
    scaler = StandardScaler().fit(train_df.iloc[tr][FEATURE_COLS])
    X_tr, y_tr = scaler.transform(train_df.iloc[tr][FEATURE_COLS]), train_df.iloc[tr]['target'].values
    X_va, y_va = scaler.transform(train_df.iloc[va][FEATURE_COLS]), train_df.iloc[va]['target'].values

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.001, random_state=SEED)
    train_loss, val_loss = [], []
    for _ in range(epochs):
        mlp.partial_fit(X_tr, y_tr, classes=[0, 1])
        train_loss.append(log_loss(y_tr, mlp.predict_proba(X_tr)[:, 1], labels=[0, 1]))
        val_loss.append(log_loss(y_va, mlp.predict_proba(X_va)[:, 1], labels=[0, 1]))

    pd.DataFrame({'epoch': range(1, epochs + 1), 'train_loss': train_loss, 'val_loss': val_loss}) \
        .to_csv(MODELS_DIR / "history_Multi_Layer_Perceptron.csv", index=False)
    plot_history("Multi_Layer_Perceptron", train_loss, val_loss, "Epoch")


def plot_history(name, train_loss, val_loss, xlabel):
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='validation loss')
    plt.xlabel(xlabel)
    plt.ylabel('Log loss')
    plt.title(f"{name} training history")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_PLOTS_DIR / f"loss_curve_{name}.png", dpi=150)
    plt.close()


def main():
    for d in [MODELS_DIR, MODEL_PLOTS_DIR, PREDICTIONS_DIR]:
        os.makedirs(d, exist_ok=True)

    train_df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_df = pd.read_parquet(FEATURES_DIR / "student_test_features.parquet")
    arcs_by_trip = load_arcs_by_trip(TRAIN_DIR, train_df['trip_id'].unique())
    arcs_by_trip.update(load_arcs_by_trip(STUDENT_TEST_DIR, test_df['trip_id'].unique()))

    fold_rows, summary_rows = [], []
    for name, spec in get_model_catalog().items():
        print(f"\n=== {name} ===")
        folds = cross_validate(name, spec['model'], train_df, arcs_by_trip)
        fold_rows += folds

        # refit on all labelled rows of the 80 training trips
        start = time.perf_counter()
        fit_df = labelled_only(train_df)
        model = clone(spec['model']).fit(fit_df[FEATURE_COLS], fit_df['target'])
        train_time = time.perf_counter() - start

        # training-set score, compared with CV to judge over/underfitting
        train_metrics, _ = evaluate(train_df, model.predict_proba(train_df[FEATURE_COLS])[:, 1], arcs_by_trip)

        start = time.perf_counter()
        test_probs = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
        infer_time = time.perf_counter() - start
        test_metrics, _ = evaluate(test_df, test_probs, arcs_by_trip)

        save_confusion_matrix(name, test_df, test_probs)
        test_df[['trip_id', 'obs_idx', 'arc_id', 'true_arc', 'target']].assign(prob=test_probs) \
            .to_parquet(PREDICTIONS_DIR / f"{name}_student_test.parquet", index=False)
        if name == 'Hist_Gradient_Boosting':
            save_hgb_history(model)

        cv = pd.DataFrame(folds)
        summary_rows.append({
            'model': name,
            'family': spec['family'],
            'config': spec['config'],
            'cv_macro_f1_mean': round(cv['macro_f1'].mean(), 4),
            'cv_macro_f1_std': round(cv['macro_f1'].std(), 4),
            'cv_top1_mean': round(cv['top1_accuracy'].mean(), 4),
            'cv_log_loss_mean': round(cv['log_loss'].mean(), 4),
            'train_macro_f1': train_metrics['macro_f1'],
            'train_top1': train_metrics['top1_accuracy'],
            **{f'test_{k}': v for k, v in test_metrics.items()},
            'train_time_s': round(train_time, 2),
            'inference_time_s': round(infer_time, 3),
            'inference_ms_per_obs': round(infer_time / test_metrics['observations'] * 1000, 4),
        })
        print(f"  student_test: macro F1={test_metrics['macro_f1']:.4f}, top-1={test_metrics['top1_accuracy']:.4f}")

    print("\n=== MLP epoch-wise loss history ===")
    mlp_loss_history(train_df)

    pd.DataFrame(fold_rows).to_csv(MODELS_DIR / "cv_folds.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    with open(MODELS_DIR / "run_config.json", 'w') as f:
        json.dump({'seed': SEED, 'n_folds': N_FOLDS, 'features': FEATURE_COLS,
                   'models': {n: s['config'] for n, s in get_model_catalog().items()}}, f, indent=2)
    print("\n" + summary[['model', 'cv_macro_f1_mean', 'test_macro_f1', 'test_top1_accuracy',
                          'test_log_loss', 'test_route_continuity']].to_string(index=False))


if __name__ == "__main__":
    main()
