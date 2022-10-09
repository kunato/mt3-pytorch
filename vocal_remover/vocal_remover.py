import os
import librosa
import numpy as np
import torch

from .lib import nets
from .lib import spec_utils

from .inference import Separator

class VocalRemover:
    def __init__(self, pretrained_model = 'models/baseline.pth', 
            sr = 44100, n_fft = 2048, hop_length = 1024, batchsize = 4, 
            cropsize = 256, tta = False, postprocess = False):

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.tta = tta
        self.postprocess = postprocess

        self.device = torch.device('cpu')
        self.model = nets.CascadedNet(self.n_fft, 32, 128)
        self.model.load_state_dict(torch.load(pretrained_model, map_location=self.device))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)

    def predict(self, audio, audio_sr):
        X = librosa.resample(audio, orig_sr=audio_sr, target_sr=self.sr)

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        X_spec = spec_utils.wave_to_spectrogram(X, self.hop_length, self.n_fft)

        sp = Separator(self.model, self.device, self.batchsize, self.cropsize, self.postprocess)

        if self.tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=self.hop_length)

        return wave, self.sr