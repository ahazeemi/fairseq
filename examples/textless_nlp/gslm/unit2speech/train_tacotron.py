import argparse
import logging
import time

from examples.textless_nlp.gslm.unit2speech.tts_data import (
    TacotronInputDataset,
)
from examples.textless_nlp.gslm.unit2speech.utils import (
    load_quantized_audio_from_file,
    load_tacotron,
)
import torch
from scipy.io.wavfile import read
import numpy as np
from examples.textless_nlp.gslm.unit2speech.tacotron2 import layers, utils

from torch import nn


class TextMelLoader():
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self,  hparams):
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Training tacotron to work with discrete speech units. "
    )
    parser.add_argument(
        "--quantized_unit_path",
        type=str,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--acoustic_model_path",
        type=str,
        help="Acoustic model file path to use for inference",
    )
    parser.add_argument("--max_decoder_steps", type=int, default=2000)

    return parser


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def process_batch(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return 0, input_lengths, mel_padded, gate_padded, \
            output_lengths


def main(args, logger):
    # Load quantized audio
    logger.info(f"Loading quantized audio from {args.quantized_unit_path}...")
    quantized_unit_file = args.quantized_unit_path

    logger.info(f"Loading acoustic model from {args.acoustic_model_path}...")
    tacotron_model, sample_rate, hparams = load_tacotron(
        tacotron_model_path=args.acoustic_model_path,
        max_decoder_steps=args.max_decoder_steps,
    )

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(
        tacotron_model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay
    )

    iteration = 0
    epoch_offset = 0

    tacotron_model.train()
    is_overflow = False

    criterion = Tacotron2Loss()

    tacotron_dataset = TacotronInputDataset(hparams)

    names_batch, quantized_units_batch = load_quantized_audio_from_file(
        file_path=quantized_unit_file
    )

    batch_processor = TextMelCollate(n_frames_per_step=hparams.n_frames_per_step)

    loader = TextMelLoader(hparams)

    batch = []
    for name, quantized_units in zip(names_batch, quantized_units_batch):
        quantized_units_str = " ".join(map(str, quantized_units))
        mel = loader.get_mel('/content/justwavfiles/' + name)
        batch.append([quantized_units_str, mel])

    _, mel_padded, gate_padded, _, _ = batch_processor.process_batch(batch[:1])

    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))

        for name, quantized_units, mel, gate in zip(names_batch, quantized_units_batch, mel_padded, gate_padded):

            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            quantized_units_str = " ".join(map(str, quantized_units))
            tacotron_input = tacotron_dataset.get_tensor(quantized_units_str)

            tacotron_model.zero_grad()

            y = [mel.cuda(), gate.cuda()]

            model_output = tacotron_model.inference(tacotron_input.unsqueeze(0).cuda(), None, ret_has_eos=False)

            loss = criterion(model_output, y)

            reduced_loss = loss.item()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                tacotron_model.parameters(), hparams.grad_clip_thresh
            )

            optimizer.step()

            if not is_overflow:
                duration = time.perf_counter() - start
                print(
                    "Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                        iteration, reduced_loss, grad_norm, duration
                    )
                )
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration
                )

            iteration += 1


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
