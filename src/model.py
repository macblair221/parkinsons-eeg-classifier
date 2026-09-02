import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


def train_evaluation_pipeline(df):
    '''
    Takes the extracted EEG features, splits them into training and test sets, trains a Random Forest
    classifier, and evaluates its performance.
    '''
    print("\n===== Initializing ML Pipeline ======")

    X = df[['plv_beta_c3_c4', 'pac_c3','pac_c4','theta_power_c3',
            'theta_power_c4','beta_power_c3',
            'beta_power_c4']]
    
    y = df['medication_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} epochs, Testing on {len(X_test)} epochs.")

    # Initialize and train RF
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
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


def loso_evaluation(df):
    '''
    Leave-One-Subject-Out: train on all subjects but one, test on the held-out
    subject. No subject appears in both train and test, so the score reflects
    decoding of medication state rather than recognition of individuals.
    '''
    FEATURES = ['plv_beta_c3_c4', 'pac_c3', 'pac_c4', 'theta_power_c3',
                'theta_power_c4', 'beta_power_c3', 'beta_power_c4']
    X, y, groups = df[FEATURES], df['medication_state'], df['subject_id']

    aucs = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        held_out = groups.iloc[test_idx].iloc[0]
        if y.iloc[test_idx].nunique() < 2:
            print(f"sub-{held_out}: skipped (only one class present)")
            continue
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = rf.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y.iloc[test_idx], prob)
        aucs.append(auc)
        print(f"sub-{held_out}: ROC-AUC {auc:.3f}")

    print(f"\nLOSO mean ROC-AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} "
          f"over {len(aucs)} subjects")
    return aucs
