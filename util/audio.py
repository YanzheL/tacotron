import librosa.filters
import scipy as sp
import librosa
import librosa.filters
import tensorflow as tf
import numpy as np
from hparams import hparams
from scipy.signal import lfilter


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    sp.io.wavfile.write(path, hparams.sample_rate, wav.astype(sp.int16))


def preemphasis(x):
    return lfilter([1, -hparams.preemphasis], [1], x)


def preemphasis_tensorflow(x):
    # return tf.tensor_scatter_sub(x, [1,2,3,4], hparams.preemphasis * x[:-1])
    return tf.concat([[x[0]], x[1:] - hparams.preemphasis * x[:-1]], 0)
    # return x[1:] - hparams.preemphasis * x[:-1]
    # def l(h, x):
    #     y=x.copy()
    #     y[1:] = x[1:] + h * x[:-1]
    #     return y
    #
    # def r(h, x):
    #     y = x.copy()
    #     y[1:] = x[1:] - h * y[:-1]
    #     return y


def inv_preemphasis(x):
    # print("inv_preemphasis(x), x: shape={}".format(tf.shape(x)))
    return lfilter([1], [1, -hparams.preemphasis], x)


def inv_preemphasis_tensorflow(x):
    return x[1:] - hparams.preemphasis * x[:-1]


def spectrogram(y):
    D = _stft(y)
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def spectrogram_tensorflow(y):
    D = _stft_tensorflow(y)
    S = _amp_to_db_tensorflow(tf.abs(D)) - hparams.ref_level_db
    return _normalize_tensorflow(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** hparams.power))  # Reconstruct phase


def melspectrogram(y):
    D = _stft(y)
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    return _normalize(S)


def melspectrogram_tensorflow(y):
    D = _stft_tensorflow(y)
    S = _amp_to_db_tensorflow(_linear_to_mel_tensorflow(tf.abs(D))) - hparams.ref_level_db
    return _normalize_tensorflow(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(hparams.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x + window_length]) < threshold:
            return x + hop_length
    return len(wav)


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = sp.exp(2j * sp.pi * sp.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hparams.griffin_lim_iters):
        angles = sp.exp(1j * sp.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    ret = sp.dot(_mel_basis, spectrogram)
    # print("_mel_basis={},spectrogram={},ret={}".format(
    #     _mel_basis.shape,
    #     spectrogram.shape,
    #     ret.shape
    # ))
    return ret


def _linear_to_mel_tensorflow(spectrogram):
    A = tf.signal.linear_to_mel_weight_matrix(
        num_spectrogram_bins=hparams.num_freq,
        num_mel_bins=hparams.num_mels,
        sample_rate=hparams.sample_rate,
        dtype=tf.dtypes.float32
    )
    M = tf.tensordot(spectrogram, A, 1)
    # print("A={},spectrogram={},M={}".format(
    #     tf.shape(A),
    #     tf.shape(spectrogram),
    #     tf.shape(M)
    # ))
    return M


def _build_mel_basis():
    n_fft = (hparams.num_freq - 1) * 2
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)


def _amp_to_db(x):
    return 20 * sp.log10(sp.maximum(1e-5, x))


def _db_to_amp(x):
    return sp.power(10.0, x * 0.05)


def _normalize(S):
    return sp.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (sp.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


def inv_spectrogram_tensorflow(spectrogram):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

    Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
    inv_preemphasis on the output after running the graph.
    '''
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hparams.ref_level_db)
    return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def _griffin_lim_tensorflow(S):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex)
        for i in range(hparams.griffin_lim_iters):
            est = _stft_tensorflow(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles)
        return tf.squeeze(y, 0)


def _istft_tensorflow(stfts):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


# Conversions:


def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _amp_to_db_tensorflow(x):
    return 20 * (tf.log(tf.maximum(1e-5, x)) / tf.log(10.0))


def _normalize_tensorflow(S):
    return tf.clip_by_value((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize_tensorflow(S):
    return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
