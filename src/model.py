import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def feature_columns(df):
    return [c for c in df.columns if c not in ('medication_state', 'subject_id')]


def train_evaluation_pipeline(df):
    '''
    Takes the extracted EEG features, splits them into training and test sets, trains a Random Forest
    classifier, and evaluates its performance.
    '''
    print("\n===== Initializing ML Pipeline ======")

    X = df[feature_columns(df)]
    
    y = df['medication_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} epochs, Testing on {len(X_test)} epochs.")

    # Initialize and train RF
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)


    # Generate predictions on test set
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:,1] # Probabilities for AUC

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n===== Model Evaluation =====")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"ROC-AUC:   {auc:.3f}\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance 
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(importances.to_string(index=False))

    return rf_model


def loso_evaluation(df, verbose = True):
    '''
    Leave-One-Subject-Out: train on all subjects but one, test on the held-out
    subject. No subject appears in both train and test, so the score reflects
    decoding of medication state rather than recognition of individuals.
    '''
    FEATURES = feature_columns(df)
    X, y, groups = df[FEATURES], df['medication_state'], df['subject_id']

    aucs = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        held_out = groups.iloc[test_idx].iloc[0]
        if y.iloc[test_idx].nunique() < 2:
            print(f"sub-{held_out}: skipped (only one class present)")
            continue
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = rf.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y.iloc[test_idx], prob)
        aucs.append(auc)
        if verbose:
            print(f"sub-{held_out}: ROC-AUC {auc:.3f}")

    if verbose:
        print(f"\nLOSO mean ROC-AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} "
              f"over {len(aucs)} subjects")
    return aucs

def shuffled_label_control(df, n_repeats=5, seed=0, observed=None):
    '''
    Negative control: shuffle medication_state WITHIN each subject, then run the
    same LOSO evaluation. Shuffling within-subject preserves every property of
    the data except the link between features and label, so any score above
    chance here is produced by the evaluation itself rather than by neural signal.
    A correct pipeline should land at ~0.50.
    '''
    rng = np.random.default_rng(seed)
    means = []

    for rep in range(n_repeats):
        shuffled = df.copy()
        # Permute labels inside each subject, so per-subject class balance is preserved
        shuffled['medication_state'] = (
            shuffled.groupby('subject_id')['medication_state']
                    .transform(lambda s: rng.permutation(s.values))
        )
        aucs = loso_evaluation(shuffled, verbose=False)
        means.append(np.mean(aucs))

    print(f"\n=== CONTROL: shuffled-label LOSO mean {np.mean(means):.3f} "
        f"+/- {np.std(means):.3f} across {n_repeats} repeats ===")

    if observed is not None:
        n_ge = sum(1 for m in means if m >= observed)
        p = (1 + n_ge) / (1 + n_repeats)
        print(f"Observed {observed:.3f} vs null: {n_ge}/{n_repeats} permutations "
              f"at or above. Permutation p = {p:.4f}")
    return means
