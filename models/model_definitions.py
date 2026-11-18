# Defines the five candidate classifiers from different algorithm families
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd

from config import SEED, MODELS_DIR


def best_model_name():
    # primary selection metric: grouped-CV Macro F1 on the 80 train trips (log loss breaks ties)
    df = pd.read_csv(MODELS_DIR / "model_comparison.csv")
    return df.sort_values(['cv_macro_f1_mean', 'cv_log_loss_mean'], ascending=[False, True])['model'].iloc[0]


def get_model_catalog():
    return {
        'Logistic_Regression': {
            'family': 'Linear & Probabilistic',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=500, random_state=SEED, class_weight='balanced'))
            ]),
            'config': 'C=1.0, max_iter=500, balanced class weights'
        },
        'Distance_Weighted_KNN': {
            'family': 'Instance-Based',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1))
            ]),
            'config': 'k=7, weights=distance, euclidean metric'
        },
        'Random_Forest': {
            'family': 'Ensemble (Bagging)',
            'model': RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=10,
                                            class_weight='balanced', random_state=SEED, n_jobs=-1),
            'config': 'n_estimators=100, max_depth=12, min_samples_split=10, balanced'
        },
        'Hist_Gradient_Boosting': {
            'family': 'Ensemble (Boosting)',
            # early stopping keeps an internal validation split, giving a per-iteration loss history
            'model': HistGradientBoostingClassifier(max_iter=100, max_depth=8, learning_rate=0.1,
                                                    early_stopping=True, random_state=SEED),
            'config': 'learning_rate=0.1, max_iter=100, max_depth=8, early_stopping'
        },
        'Multi_Layer_Perceptron': {
            'family': 'Neural Networks',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.001,
                                      max_iter=150, early_stopping=True, random_state=SEED))
            ]),
            'config': 'hidden_layers=(64, 32), relu, adam, alpha=0.001, early_stopping'
        },
    }
