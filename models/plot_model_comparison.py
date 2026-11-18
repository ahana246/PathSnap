# Plots CV and student_test metrics of all models from model_comparison.csv and cv_folds.csv
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import MODELS_DIR, MODEL_PLOTS_DIR


def bar_plot(df, cols, labels, title, ylabel, filename, errors=None):
    ax = df.set_index('model')[cols].plot.bar(figsize=(8, 4.5), yerr=errors, capsize=4, rot=20)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(labels)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_PLOTS_DIR / filename, dpi=150)
    plt.close()


def main():
    os.makedirs(MODEL_PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(MODELS_DIR / "model_comparison.csv")
    folds = pd.read_csv(MODELS_DIR / "cv_folds.csv")

    # CV macro F1 with fold standard deviation as error bars
    bar_plot(df, ['cv_macro_f1_mean'], ['CV macro F1'], 'Cross-validation Macro F1 (5 grouped folds)',
             'Macro F1', 'cv_macro_f1.png', errors=df['cv_macro_f1_std'].values)

    bar_plot(df, ['test_top1_accuracy', 'test_macro_f1'], ['Top-1 accuracy', 'Macro F1'],
             'Student-test Top-1 Accuracy and Macro F1', 'Score', 'student_test_top1_f1.png')

    bar_plot(df, ['cv_log_loss_mean', 'test_log_loss'], ['CV', 'Student test'],
             'Log Loss (lower is better)', 'Log loss', 'log_loss.png')

    bar_plot(df, ['train_macro_f1', 'cv_macro_f1_mean', 'test_macro_f1'], ['Train', 'CV', 'Student test'],
             'Train vs CV vs Student-test Macro F1 (overfitting check)', 'Macro F1', 'overfitting_check.png')

    bar_plot(df, ['test_route_continuity'], ['Route continuity'], 'Student-test Route Continuity',
             'Share of connected consecutive predictions', 'route_continuity.png')

    bar_plot(df, ['inference_ms_per_obs'], ['Inference time'], 'Inference time per GPS observation',
             'ms / observation', 'inference_time.png')

    # spread of macro F1 across folds shows model stability
    ax = folds.boxplot(column='macro_f1', by='model', figsize=(8, 4.5), rot=20)
    ax.set_title('Macro F1 across CV folds')
    ax.set_ylabel('Macro F1')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(MODEL_PLOTS_DIR / "cv_fold_stability.png", dpi=150)
    plt.close()
    print(f"[+] Saved comparison plots to {MODEL_PLOTS_DIR}")


if __name__ == "__main__":
    main()
