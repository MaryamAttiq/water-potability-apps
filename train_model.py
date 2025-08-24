"""
Train a RandomForest on the Water Potability dataset with class balancing (oversampling or undersampling).
Saves the best model to model.pkl and writes metrics.json + confusion_matrix.png

Usage examples:
  python train_model.py --csv water_potability.csv --sampling over
  python train_model.py --csv water_potability.csv --sampling under --scoring roc_auc --n_iter 40
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # Standard Kaggle columns: 'ph','Hardness','Solids','Chloramines','Sulfate','Conductivity',
    # 'Organic_carbon','Trihalomethanes','Turbidity','Potability'
    if 'Potability' not in df.columns:
        raise ValueError("Target column 'Potability' not found in CSV. Make sure your file has this column.")
    X = df.drop(columns=['Potability'])
    y = df['Potability'].astype(int)
    return X, y

def build_pipeline(sampling: str, random_state: int = 42):
    sampler = RandomOverSampler(random_state=random_state) if sampling == 'over' else RandomUnderSampler(random_state=random_state)
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    pipeline = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # handle NaNs safely
        ('sampler', sampler),
        ('model', rf),
    ])
    return pipeline

def param_distributions():
    return {
        'model__n_estimators': [200, 400, 600, 800, 1000],
        'model__max_depth': [None, 8, 12, 16, 24, 32],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None],
        'model__bootstrap': [True, False],
        'model__criterion': ['gini', 'entropy'],
    }

def evaluate(model, X_test, y_test, out_dir: str):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, str(z), ha='center', va='center')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(cm_path)
    plt.close(fig)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics, cm_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='water_potability.csv', help='Path to water_potability.csv')
    parser.add_argument('--sampling', type=str, choices=['over','under'], default='over', help='Class balancing technique')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=25, help='RandomizedSearch iterations')
    parser.add_argument('--scoring', type=str, default='f1', help='Score for model selection (e.g., f1, roc_auc, accuracy)')
    parser.add_argument('--out_model', type=str, default='model.pkl')
    parser.add_argument('--out_dir', type=str, default='outputs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, y = load_data(args.csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = build_pipeline(args.sampling, random_state=args.seed)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions(),
        n_iter=args.n_iter,
        scoring=args.scoring,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed),
        n_jobs=-1,
        refit=True,
        random_state=args.seed,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Save model
    model_path = os.path.join(args.out_dir, args.out_model)
    joblib.dump(best_model, model_path)

    # Evaluate
    metrics, cm_path = evaluate(best_model, X_test, y_test, args.out_dir)

    # Save feature importances
    rf = best_model.named_steps['model']
    importances = rf.feature_importances_
    fi = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
    fi_path = os.path.join(args.out_dir, 'feature_importances.csv')
    fi.to_csv(fi_path, index=False)

    summary = {
        'best_params': search.best_params_,
        'best_score_cv': search.best_score_,
        'metrics_path': os.path.join(args.out_dir, 'metrics.json'),
        'confusion_matrix_path': cm_path,
        'feature_importances_path': fi_path,
        'model_path': model_path
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import os
    main()
