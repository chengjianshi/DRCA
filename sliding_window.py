import numpy as np
from sklearn.preprocessing import normalize


def sliding_window(
    signal: np.ndarray,
    frequency: int,
    ws: int,
    overlapping: float =0.,
    vmc=False,
    norm=True,
    center=True,
):
    '''
    Return a sliding window over ndarray signal in any number of dimensions.

    input signal shape: (data_points, features)
    output features shape: (n_steps, seq_len, features)

    params:
        signal - numpy ndarray
        frequency - collected frequency of input signal
        ws - an int representing the window size in secs, 
            consistant in all dimension of signal
        ol - overlapping percentage on sliding window, default 0
        vmc - vector magnitude count representation default False
        norm - normalize signal in each dim between [-1,1], default True
        Smoothing - smoothing signal via moving-average
        center - centering siganl in each dim, default true
    '''

    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal has to be ndarray.")

    if vmc:
        signal = np.sqrt(np.sum(signal**2, axis=1)).reshape(len(signal), -1)

    leng, n_features = signal.shape

    # number of window slices, truncate undivadable part
    seq_len = ws * frequency  # number of data points in each window

    if ws > leng:
        raise TypeError(f"window size {ws} larger than given signal length {leng}")

    step_size = int(seq_len * (1 - overlapping))
    n_steps = leng // step_size - 1

    sequences = [signal[i * step_size:i * step_size + seq_len, :]
                 for i in range(n_steps)]

    # sequences shape: (n_steps, seq_len, n_features)
    if norm:
        sequences = [(sequences[i] - np.min(sequences[i], axis=0)) /
                     (np.max(sequences[i], axis=0) -
                      np.min(sequences[i], axis=0))
                     for i in range(n_steps)]
        sequences = [normalize(sequences[i], axis=0) for i in range(n_steps)]

    if center:
        sequences = [sequences[i] - np.mean(sequences[i], axis=0)
                     for i in range(n_steps)]

    return np.array(sequences, dtype=np.float32)
