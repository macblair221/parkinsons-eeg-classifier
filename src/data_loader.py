import os
import openneuro
import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

from preprocessing import clean_and_epoch_data
from features import extract_neural_features
from model import train_evaluation_pipeline, loso_evaluation, shuffled_label_control
import pandas as pd


def download_dataset(dataset_id="ds002778", target_dir='./data/raw'):
    '''
    Pulls the OpenNeuro dataset if it doesn't already locally exist
    '''

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Downloading {dataset_id} to {target_dir}...")
        openneuro.download(dataset=dataset_id, target_dir=target_dir)
    else:
        print(f"Dataset already exists at {target_dir}")
    
    return target_dir


def load_subject_data(bids_root, subject_id):
    ''''
    Loads both the medicated (on) and unmedicated (off) resting state sessions for a given subject.
    Returns: dictionary containing raw MNE objects.
    '''

    sessions = ['on', 'off']
    task = 'rest'
    subject_data = {}

    for ses in sessions:
        try:
            bids_path = BIDSPath(subject=subject_id, session=ses, task=task, datatype='eeg', root=bids_root)

            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            raw.load_data()

            subject_data[ses] = raw
            print(f"Successfully loaded sub-{subject_id} ses-{ses}")

        except FileNotFoundError:
            print(f"Warning: sub-{subject_id} ses-{ses} not found. Skipping.")
            subject_data[ses] = None

    return subject_data

if __name__ == "__main__":
    # Ensure data is downloaded
    dataset_path = download_dataset()
    
    # Discover all subjects in the downloaded BIDS directory
    # -- Prevents hardcoding patient IDs and makes the script reproducable
    all_subjects = get_entity_vals(dataset_path, 'subject')

    # Filter the list to only inlcude PD patients (No control subjects)
    pd_subjects = [sub for sub in all_subjects if sub.startswith('pd')]
    print(f"Found {len(pd_subjects)} subjects total.")

   
    test_subjects = pd_subjects

    all_features = []

    for sub in test_subjects:
        print(f"\n==== Loading Subjects: {sub} ====")
        subject_data = load_subject_data(bids_root=dataset_path, subject_id=sub)


        # OFF state (label = 0)
        if subject_data['off'] is not None:
            epochs_off = clean_and_epoch_data(subject_data['off'])
            if epochs_off:
                df_off = extract_neural_features(epochs_off, label_state=0, subject_id=sub)
                all_features.append(df_off)
                print(f"Extracted features for {len(df_off)} OFF epochs.")

        # ON state (label = 1)
        if subject_data['on'] is not None:
            epochs_on = clean_and_epoch_data(subject_data['on'])
            if epochs_on:
                df_on = extract_neural_features(epochs_on, label_state=1, subject_id=sub)
                all_features.append(df_on)
                print(f"Extracted features for {len(df_on)} ON epochs.")

    # Combine everything into one dataset
    final_dataset = pd.concat(all_features, ignore_index=True)
    final_dataset.to_csv('features.csv', index=False)
    print(final_dataset.head())
    print(f"shape: {final_dataset.shape}")

    # Run the ML pipeline
    trained_model = train_evaluation_pipeline(final_dataset)

    # Same features, leakage-free split — the honest number
    loso_evaluation(final_dataset)

    # Negative control: same pipeline, labels destroyed
    shuffled_label_control(final_dataset)

