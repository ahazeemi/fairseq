# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import search, utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerModel
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor


class SimpleSequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        retain_dropout=False,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
    ):
        """Generates translations of a given source sentence.
        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        self.model = EnsembleModel(models)
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        if not self.retain_dropout:
            self.model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(self, sample: Dict[str, Dict[str, Tensor]]):
        """Generate translations."""
        return self._generate(sample)

    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, sample: Dict[str, Dict[str, Tensor]]):
        """Generate translations."""
        return self._generate(sample)

    def _generate(self, sample: Dict[str, Dict[str, Tensor]]):

        encoder_input: Dict[str, Tensor] = {}
        for k, v in sample["net_input"].items():
            if k != "prev_output_tokens":
                encoder_input[k] = v

        src_tokens = encoder_input["src_tokens"]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        )
        # bsz: total number of sentences in beam
        bsz, src_len = src_tokens.size()

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = min(self.beam_size, self.vocab_size - 1)
        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(
            src_tokens=encoder_input["src_tokens"],
            src_lengths=encoder_input["src_lengths"],
        )

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens.data)
        )  # +1 for eos; pad is never choosed for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .fill_(self.pad)
            .to(src_tokens.data)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                self.model.reorder_incremental_state(reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            lprobs, avg_attn_scores = self.model.forward_decoder(
                tokens[:, : step + 1], encoder_outs, self.temperature
            )

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)
            if step < max_len:
                self.search.set_src_lengths(src_lengths)
                cand_scores, cand_indices, cand_beams = self.search.step(
                    step,
                    lprobs.view(bsz, -1, self.vocab_size),
                    scores.view(bsz, beam_size, -1)[:, :, :step],
                )
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                eos_scores, eos_bbsz_idx = torch.sort(
                    lprobs[:, self.eos], descending=True
                )
                num_remaining_sent -= self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                )

                # dummy data to be consistent with true branch for type check
                cand_beams = torch.empty(0)
                cand_indices = torch.empty(0)
                cand_scores = torch.empty(0)
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(
                (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
            )

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)

            # only consider eos when it's among the top beam_size indices
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )
                num_remaining_sent -= self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                )

            if num_remaining_sent == 0:
                break

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            _, active_hypos = torch.topk(active_mask, k=beam_size, dim=1, largest=False)

            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )

            scores[:, :step] = torch.index_select(
                scores[:, :step], dim=0, index=active_bbsz_idx
            )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            # make into beam container
            BCList = [BeamContainer(elem["score"], elem) for elem in finalized[sent]]
            BCList.sort()
            BCList.reverse()
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], [x.elem for x in BCList]
            )

        return finalized

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        Returns number of sentences being finalized.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS
        tokens_clone[:, step] = self.eos

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty
        # set() is not supported in script export
        sents_seen: Dict[int, Optional[Tensor]] = {}
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            sent = idx // beam_size
            if sent not in sents_seen:
                sents_seen[sent] = None
            # sents_seen.add(sent)

            if len(finalized[sent]) < beam_size:
                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": torch.empty(0),  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished = 0
        for sent in sents_seen.keys():
            # check termination conditions for this sentence
            if not finished[sent] and len(finalized[sent]) == beam_size:
                finished[sent] = True
                newly_finished += 1
        return newly_finished

    def _decode(self, tokens, encoder_out: EncoderOut, temperature: float = 1.0):
        if self.incremental_states is not None:
            decoder_out = self.model.decoder.forward(
                tokens,
                encoder_out=encoder_out,
                incremental_state=self.incremental_states,
            )
        else:
            decoder_out = self.model.decoder.forward(tokens, encoder_out=encoder_out)
        tep = (decoder_out[0][:, -1, :].div_(temperature), decoder_out[1])
        probs = self.model.get_normalized_probs(tep, log_probs=True)
        return probs


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]]

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Optional[Dict[str, Dict[str, Optional[Tensor]]]], {})
                for i in range(len(self.models))
            ],
        )
        self.has_incremental: bool = False
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, src_tokens, src_lengths):
        if not self.has_encoder():
            return None
        return [
            model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            for model in self.models
        ]

    @torch.jit.export
    def forward_decoder(
        self, tokens, encoder_outs: List[EncoderOut], temperature: float = 1.0
    ):

        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[EncoderOut] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[i]
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
            # Attention is not used in this version of sequence generator. And
            # variable type in the torchscript need to be static. So attention part
            # is commented out. Currently only supporting Dict type output
            attn = decoder_out[1]["attn"][0]
            if attn is not None:
                attn = attn[:, -1, :]
            # if isinstance(attn_holder, dict):
            #     attn = attn_holder.get('attn', None)
            # if isinstance(attn, list):
            #     attn = attn_holder[0]
            # if attn is not None:
            #     attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1, :].div_(temperature),
                decoder_out[1],
            )
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )

            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Optional[List[EncoderOut]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[EncoderOut] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(self, new_order):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state(
                self.incremental_states[i], new_order
            )

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs to support TransformerModel
    @torch.jit.export
    def get_normalized_probs(
        self,
        model: TransformerModel,
        net_output: Tuple[Tensor, Dict[str, List[Optional[Tensor]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(model, "decoder"):
            if (
                hasattr(model.decoder, "adaptive_softmax")
                and model.decoder.adaptive_softmax is not None
            ):
                if sample is not None:
                    assert "target" in sample
                    target = sample["target"]
                else:
                    target = None
                out = model.decoder.adaptive_softmax.get_log_prob(
                    net_output[0], target=target
                )
                return out.exp_() if not log_probs else out

            logits = net_output[0]
            if log_probs:
                return utils.log_softmax(
                    logits, dim=-1, onnx_trace=model.decoder.onnx_trace
                )
            else:
                return utils.softmax(
                    logits, dim=-1, onnx_trace=model.decoder.onnx_trace
                )
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)


@torch.jit.script
class BeamContainer(object):
    def __init__(self, score: float, elem: Dict[str, Tensor]):
        self.score = score
        self.elem = elem

    def __lt__(self, other):
        # type: (BeamContainer) -> bool
        # Due to https://github.com/pytorch/pytorch/issues/20388,
        # this has to use old style type annotations
        return self.score < other.score