# Parkinson's Disease Medication-State Classifier

Decoding levodopa medication state (ON vs. OFF) from resting-state EEG, with an emphasis on
evaluation methodology: what the data can actually support, and what a careless pipeline would
have claimed instead.

**Headline result: 0.578 ROC-AUC on held-out patients** (13 subjects, leave-one-subject-out),
against a shuffled-label null of 0.500 ± 0.012 (permutation p = 0.0099, 100 permutations).

An earlier version of this project reported 0.68 from a random train/test split. That number was
wrong, and most of this README is about why.

---

## Results

| Evaluation | ROC-AUC | Notes |
|---|---|---|
| Pooled random split | 0.796 | **Leaks.** Epochs from the same patient in train and test. |
| Leave-one-subject-out, 15 subjects | 0.583 ± 0.115 | No subject appears in both train and test. |
| Leave-one-subject-out, 13 subjects | 0.578 ± 0.087 | Two provenance-flagged subjects excluded (below). |
| Shuffled-label null, 100 permutations | 0.500 ± 0.012 | Labels permuted within subject. |
| Permutation p-value | **0.0099** | 0/100 permutations reached the observed score. |

The gap between the first row and the third — 0.796 vs 0.578 — is the point of this project.
Both numbers come from the same features and the same model. Only the split differs.

---

## Why the pooled split is wrong

The original pipeline pooled every subject's epochs into one table and called
`train_test_split(..., test_size=0.2, stratify=y)`. That splits at the *epoch* level, so 2-second
windows from the same continuous recording land in both training and test sets. Resting EEG is
strongly autocorrelated, so adjacent epochs are near-duplicates: the model can memorise a
patient's individual spectral signature and score well without decoding anything about medication.

The fix is `LeaveOneGroupOut` grouped on `subject_id`. Each fold holds out one patient entirely.
Both evaluations are kept in the codebase deliberately — the difference between them is a
measurement, not an embarrassment.

Note that "within-subject design" describes this dataset's *experimental* structure (each patient
contributes paired ON and OFF sessions and serves as their own control). It says nothing about the
validation scheme, and conflating the two is what produced the original error.

---

## Two amplitude confounds, found and removed

ON and OFF are always **separate recordings**, so session identity is perfectly confounded with the
label. Any feature sensitive to recording conditions — electrode impedance, referencing, amplifier
gain — can be read by the classifier as "medication state."

**1. Absolute → relative band power.** The original features were `log10(var(x))` per band, which
scales directly with session gain. Replacing them with each band's *share* of total power cancels
any factor that scales all bands together.

Effect on leave-one-subject-out (C3/C4 only, 7 features):

| | mean | std dev | subjects below chance |
|---|---|---|---|
| Absolute power | 0.555 | 0.188 | 8 / 15 |
| Relative power | 0.562 | 0.095 | 4 / 15 |

The mean barely moved. Everything else did: the spread halved, and the three most separable
subjects dropped sharply (1.000 → 0.693, 0.858 → 0.629, 0.840 → 0.598) while the remaining
subjects rose from 0.509 to 0.554. Inflated scores fell, suppressed scores rose — the signature of
removing an artifact rather than removing signal.

**2. Amplitude-normalised PAC.** The raw Canolty modulation index, `|mean(A_γ · e^(iφ_β))|`, scales
linearly with gamma amplitude, so it conflates coupling strength with loudness. Dividing by
`mean(A_γ)` leaves a dimensionless coupling measure in [0, 1].

This barely changed the grouped score (0.562 → 0.553) but cut the pooled score from 0.644 to
0.599, shrinking the pooled/LOSO gap from 0.082 to 0.046. Raw PAC magnitude was acting as a
subject fingerprint — useful for identifying *who*, useless for decoding *what*.

---

## Data provenance audit

Two subjects, **pd6 and pd16**, carry a note in `participants.tsv`: the ON session used
EEGLAB-preprocessed data rather than raw data. They are the only two subjects in the dataset with
any such note.

In the original run they scored **0.858 and 0.840** — two of the top three. When ON and OFF
sessions are preprocessed differently, a classifier can separate them without reference to
neurophysiology.

Retraining with both excluded, evaluated on the same remaining 13 subjects:

| | mean | std dev | below chance |
|---|---|---|---|
| Trained with pd6, pd16 | 0.563 | 0.107 | 3 / 13 |
| Trained without | 0.578 | 0.087 | 2 / 13 |

**8 of 13 subjects improved** when the flagged recordings were removed from training. Contaminated
data was not merely inflating its own scores — it was teaching the model patterns that hurt
generalisation to everyone else. The 13-subject figure is reported as the headline for this reason.

---

## Negative control

To verify that the evaluation harness itself doesn't leak, `medication_state` is permuted **within
each subject** — preserving each subject's class balance and every property of the features, and
breaking only the feature-label correspondence. A correct pipeline should score chance.

Across 100 permutations: **0.500 ± 0.012**. Zero permutations reached the observed 0.583, which is
roughly 6.9 standard deviations above the null.

---

## Method

- **Dataset:** [ds002778](https://openneuro.org/datasets/ds002778) — UC San Diego resting-state EEG,
  15 Parkinson's patients, paired ON/OFF levodopa sessions.
- **Preprocessing:** BIDS ingestion via `mne-bids`, standard 10-20 montage, 104 Hz low-pass,
  2-second fixed-length epochs. 2,992 epochs total.
- **Channels:** 7 sensorimotor electrodes (C3, Cz, C4, FC1, FC2, CP1, CP2).
- **Bands:** theta 4–8, alpha 8–13, low beta 13–20, high beta 20–30, gamma 30–50 Hz. Gamma stops
  at 50 Hz to avoid 60 Hz mains noise; beta is split because levodopa's effect differs across the
  sub-bands.
- **Features (48):** relative band power per channel per band (35), amplitude-normalised
  beta→gamma PAC per channel (7), and beta-band phase-locking across C3–C4, C3–Cz, Cz–C4 (6).
- **Model:** Random Forest, 100 trees, default depth.
- **Evaluation:** `LeaveOneGroupOut` grouped by subject; permutation control with 100 within-subject
  label shuffles.

---

## Limitations

**Relative power is compositional.** The five band shares sum to 1 by construction, so they cannot
move independently. Gamma's share falling ON medication necessarily means the other bands' combined
share rose. Band-level effects therefore cannot be interpreted in isolation — this is the cost of
the normalisation that removed the session-gain confound.

**The gamma effect may be partly muscular.** Gamma power dominates feature importance (top four
features), and gamma share is lower ON medication at every channel. The 30–50 Hz band overlaps EMG,
and PD patients OFF medication have more rigidity and tremor — so this could reflect muscle tone
rather than cortical activity. The topography argues against pure EMG: the effect peaks centrally
(C3 +0.036, C4 +0.035, Cz +0.029) and falls off at FC1 (+0.011), a 3.3× gradient in the wrong
direction for muscle artifact. But every electrode in this montage sits on the central strip, so
without peripheral channels (T7/T8, F7/F8) or ICA decomposition the question is open.

**Session is perfectly confounded with label.** ON and OFF are separate recordings for every
subject. Two amplitude-driven routes for this confound have been closed; others may remain.

**No artifact rejection.** The pipeline applies filtering and epoching only — no ICA, no
amplitude-threshold rejection, no bad-channel interpolation.

**Small N.** 15 subjects, 13 after the provenance exclusion.

**Configurations tested:** three, each motivated physiologically before being run, and all three
reported above (absolute→relative power; PAC normalisation; channel and band expansion). No
hyperparameter search was performed. This matters for interpreting the p-value.

---

## Reproducing

```bash
git clone https://github.com/macblair221/parkinsons-eeg-classifier.git
cd parkinsons-eeg-classifier
pip install -r requirements.txt

python src/data_loader.py          # downloads ds002778, extracts features -> features.csv
python src/evaluate.py             # pooled + leave-one-subject-out
python src/evaluate.py --control   # adds the 100-permutation null (slower)
```

---

## Citations

Swann, N.C., de Hemptinne, C., Aron, A.R., Ostrem, J.L., Knight, R.T. and Starr, P.A. (2015),
Elevated synchrony in Parkinson disease detected with electroencephalography. *Ann Neurol.*, 78:
742-750. https://doi.org/10.1002/ana.24507

Rockhill, A.P., Jackson, N., George, J., Aron, A., and Swann, N.C. (2020). UC San Diego Resting
State EEG Data from Patients with Parkinson's Disease. *OpenNeuro.* [Dataset]
doi: 10.18112/openneuro.ds002778.v1.0.1
