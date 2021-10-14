# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import fairseq
import soundfile as sf


class Wav2VecFeatureReader:
    """
    Wrapper class to run inference on Wav2Vec 2.0 model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        self.task = task
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.layer = layer

    def read_audio(self, fname):
        wav, sr = sf.read(fname)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.task.cfg.sample_rate, sr
        return wav

    def get_feats(self, file_path):
        x = self.read_audio(file_path)
        with torch.no_grad():
            source = torch.from_numpy(x).view(1, -1).float().cuda()
            res = self.model(
                source=source, mask=False, features_only=True, layer=self.layer
            )
            return res["layer_results"][self.layer][0].squeeze(1)
