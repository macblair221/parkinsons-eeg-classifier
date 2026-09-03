import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

# Sensorimotor strip and immediate neighbours, up from just C3/C4.
# Intersected with what the recording actually contains so a missing
# electrode degrades gracefully instead of crashing.
SENSORIMOTOR = ['C3', 'C1', 'Cz', 'C2', 'C4', 'FC1', 'FC2', 'CP1', 'CP2']

BANDS = {
    'theta':    (4.0, 8.0),
    'alpha':    (8.0, 13.0),
    'lowbeta':  (13.0, 20.0),
    'highbeta': (20.0, 30.0),
    'gamma':    (30.0, 50.0),
}


def extract_neural_features(epochs, label_state, subject_id=None):
    if epochs is None:
        return None

    wanted = [ch for ch in SENSORIMOTOR if ch in epochs.ch_names]
    picked = epochs.copy().pick(wanted)
    ch_names = picked.ch_names          # authoritative order after picking

    # Filter once per band; each array is (n_epochs, n_channels, n_times)
    band_data = {
        name: picked.copy().filter(lo, hi, verbose=False).get_data()
        for name, (lo, hi) in BANDS.items()
    }

    features_list = []

    for i in range(band_data['lowbeta'].shape[0]):
        row = {}

        # Relative power: each band's share of that channel's total power
        # across the analysed bands, so per-session gain cancels.
        powers = {name: np.var(band_data[name][i], axis=1) for name in BANDS}
        total = sum(powers.values())
        for name in BANDS:
            rel = powers[name] / total
            for idx, ch in enumerate(ch_names):
                row[f'{name}_power_{ch}'] = rel[idx]

        # Phase-locking across the motor strip, in both beta sub-bands
        for band in ('lowbeta', 'highbeta'):
            for a, b in [('C3', 'C4'), ('C3', 'Cz'), ('Cz', 'C4')]:
                if a in ch_names and b in ch_names:
                    pa = np.angle(hilbert(band_data[band][i][ch_names.index(a)]))
                    pb = np.angle(hilbert(band_data[band][i][ch_names.index(b)]))
                    row[f'plv_{band}_{a}_{b}'] = np.abs(np.mean(np.exp(1j * (pa - pb))))

        # Amplitude-normalised PAC: high-beta phase -> gamma amplitude
        for idx, ch in enumerate(ch_names):
            phase = np.angle(hilbert(band_data['highbeta'][i][idx]))
            amp = np.abs(hilbert(band_data['gamma'][i][idx]))
            row[f'pac_{ch}'] = np.abs(np.mean(amp * np.exp(1j * phase))) / np.mean(amp)

        row['medication_state'] = label_state
        row['subject_id'] = subject_id
        features_list.append(row)

    return pd.DataFrame(features_list)