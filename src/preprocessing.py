import mne

def clean_and_epoch_data(raw_data, epoch_length=2.0):
    '''
    Takes a raw MNE object, applies filtering, set the montage, and
    segments into fixed-length epochs for machine learning
    '''
    if raw_data is None:
        return None
    
    # Drop the simulus/status channel so it doesnt skew our filters
    if 'Status' in raw_data.ch_names:
        raw_data.drop_channels(['Status'])

    # Set the standard 10-20 montage 
    # -- maps the 2D channel names (c3,c4) to 3D spatial coordinates on the scalp
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_data.set_montage(montage, match_case=False, match_alias=True, on_missing='ignore')

    # Apply the lowpass filter
    raw_filtered = raw_data.filter(l_freq=None, h_freq=104.0, fir_design='firwin', verbose=False)

    # Create fixed length epochs (e.g., 2 seconds each)
    epochs = mne.make_fixed_length_epochs(raw_filtered, duration=epoch_length, preload=True, verbose=False)
   
    return epochs