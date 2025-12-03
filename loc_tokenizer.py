"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous Localization.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase



class LocTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_loc: int = 0, max_loc: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_loc: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_loc: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_loc, self.max_loc = tokenizer, bins, min_loc, max_loc

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_loc, max_loc, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "loc_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.loc_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, loc: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin input to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        loc = np.clip(loc, a_min=float(self.min_loc), a_max=float(self.max_loc))
        discretized_loc = np.digitize(loc, self.bins)

        # Handle single element vs. batch
        if len(discretized_loc.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_loc))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_loc).tolist())

    def decode_token_ids_to_loc(self, loc_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous localizations for discrete localization token IDs.

        NOTE =>> Because of the way the localizations are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_loc = self.tokenizer.vocab_size - loc_token_ids
        discretized_loc = np.clip(discretized_loc - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_loc]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
