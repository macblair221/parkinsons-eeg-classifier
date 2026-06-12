import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

def extract_neural_features(epochs, label_state):

    if epochs is None:
        return None
    
    # Filter the distinct bands
    # We stop Gamma at 50Hz to dodge 60Hz US power-line noise
    epochs_theta = epochs.copy().filter(l_freq=4.0, h_freq=6.0, verbose=False)
    epochs_beta = epochs.copy().filter(l_freq=13.0, h_freq=30.0, verbose=False)
    epochs_gamma = epochs.copy().filter(l_freq=30.0, h_freq=50.0, verbose=False)

    # Extract C3 and C4 arrays
    theta_data = epochs_theta.copy().pick(['C3', 'C4']).get_data()  
    beta_data = epochs_beta.copy().pick(['C3', 'C4']).get_data()  
    gamma_data = epochs_gamma.copy().pick(['C3', 'C4']).get_data()  

    features_list = []

    # Iterate through every 2-second epoch
    for i in range(len(beta_data)):
        # Isolate channels
        beta_c3, beta_c4 = beta_data[i][0], beta_data[i][1]
        gamma_c3, gamma_c4 = gamma_data[i][0], gamma_data[i][1]
        theta_c3, theta_c4 = theta_data[i][0], theta_data[i][1]

        # Phase Locking Value (Beta C3-C4)
        phase_beta_c3 = np.angle(hilbert(beta_c3))
        phase_beta_c4 = np.angle(hilbert(beta_c4))
        plv_beta = np.abs(np.mean(np.exp(1j * (phase_beta_c3 - phase_beta_c4))))

        # Phase Amplitude Coupling (Beta Phase -> Gamma Amplitude)
        amp_gamma_c3 = np.abs(hilbert(gamma_c3))
        amp_gamma_c4 = np.abs(hilbert(gamma_c4))

        # Simplified Canolty's Modulation Index (log transformed)
        pac_c3 = np.log10(np.abs(np.mean(amp_gamma_c3 * np.exp(1j * phase_beta_c3))))
        pac_c4 = np.log10(np.abs(np.mean(amp_gamma_c4 * np.exp(1j * phase_beta_c4))))


        # LOG Power (Theta and Beta)
        theta_power_c3 = np.log10(np.var(theta_c3))
        theta_power_c4 = np.log10(np.var(theta_c4))
        beta_power_c3 = np.log10(np.var(beta_c3))
        beta_power_c4 = np.log10(np.var(beta_c4))

        # Append to dataset
        features_list.append({
            'plv_beta_c3_c4': plv_beta,
            'pac_c3': pac_c3,
            'pac_c4': pac_c4,
            'theta_power_c3': theta_power_c3,
            'theta_power_c4': theta_power_c4,
            'beta_power_c3' : beta_power_c3,
            'beta_power_c4' : beta_power_c4,
            'medication_state': label_state

        })

    return pd.DataFrame(features_list) # Convert the list of dictionaries into a Pandas DataFrame

