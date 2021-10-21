# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


from torch import nn


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
        "--tts_model_path",
        type=str,
        help="TTS model file path to use for inference",
    )
    parser.add_argument("--max_decoder_steps", type=int, default=2000)

    return parser


def main(args, logger):
    # Load quantized audio
    logger.info(f"Loading quantized audio from {args.quantized_unit_path}...")
    quantized_unit_file = args.quantized_unit_path

    logger.info(f"Loading TTS model from {args.tts_model_path}...")
    tacotron_model, sample_rate, hparams = load_tacotron(
        tacotron_model_path=args.tts_model_path,
        max_decoder_steps=args.max_decoder_steps,
    )

    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(
        tacotron_model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay
    )

    iteration = 0
    epoch_offset = 0

    tacotron_model.train()
    is_overflow = False

    criterion = Tacotron2Loss()

    tts_dataset = TacotronInputDataset(hparams)
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))

        names_batch, quantized_units_batch = load_quantized_audio_from_file(
            file_path=quantized_unit_file
        )

        for name, quantized_units in zip(names_batch, quantized_units_batch):

            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            quantized_units_str = " ".join(map(str, quantized_units))
            tts_input = tts_dataset.get_tensor(quantized_units_str)

            tacotron_model.zero_grad()
            x, y = tacotron_model.parse_batch(tts_input)
            y_pred = tacotron_model(x)

            loss = criterion(y_pred, y)

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
