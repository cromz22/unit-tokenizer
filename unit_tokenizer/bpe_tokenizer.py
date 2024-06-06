import json
import logging

from unit_tokenizer import BaseTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class BPETokenizer(BaseTokenizer):
    """
    Pure BPE tokenizer that operates on a sequence of units.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.merges = {}
        self.swapped_merges = {}

    def _get_counts(self, units_list: list[list[int]]) -> dict[tuple[int, int], int]:
        """
        Count the number of occurrences for each pair of units within each inner list.
        """
        counts = {}
        for units in units_list:
            for pair in zip(units[:-1], units[1:]):
                if pair in counts:
                    counts[pair] += 1
                else:
                    counts[pair] = 1
        return counts

    def _merge(
        self, units_list: list[list[int]], pair: tuple[int, int], idx: int
    ) -> list[list[int]]:
        """
        Replace all occurrences of `pair` in `units_list` with `idx`.
        """
        new_units_list = []
        for units in units_list:
            new_units = []
            i = 0
            while i < len(units):
                if i < len(units) - 1 and (units[i], units[i + 1]) == pair:
                    new_units.append(idx)
                    i += 2
                else:
                    new_units.append(units[i])
                    i += 1
            new_units_list.append(new_units)
        return new_units_list

    def fit(self, train_data: list[list[int]], target_vocab_size: int) -> None:
        """
        Fit the tokenizer on `train_data`.
        """
        if not train_data or not any(train_data):
            error_message = "Training data is empty."
            self.logger.error(error_message)
            raise ValueError(error_message)

        units_list = train_data
        set_units_list = [set(units) for units in units_list]
        set_units = set.union(*set_units_list)
        initial_vocab_size = len(set_units)
        max_idx = max(set_units)

        if target_vocab_size <= initial_vocab_size:
            error_message = f"Target vocab size ({target_vocab_size}) must be greater than the initial vocab size ({initial_vocab_size})."
            self.logger.error(error_message)
            raise ValueError(error_message)

        num_merges = target_vocab_size - initial_vocab_size
        self.logger.info(f"Fitting tokenizer with {num_merges} merges.")
        self.logger.debug(f"Initial units: {units_list}")

        for i in range(num_merges):
            counts = self._get_counts(units_list)
            if not counts:
                self.logger.warning("No more pairs to merge.")
                break
            top_pair = max(counts, key=counts.get)
            new_idx = max_idx + 1
            units_list = self._merge(units_list, top_pair, new_idx)
            self.merges[top_pair] = new_idx
            self.logger.info(f"Merge {i + 1}/{num_merges}: {top_pair} -> {new_idx}")
            self.logger.debug(f"units: {units_list}")

            max_idx = new_idx

    def fit_from_file(self, train_file: str, target_vocab_size: int) -> None:
        """
        Fit the tokenizer from a file.
        `train_file` should contain a sequence of integers separated by spaces per line.
        """
        with open(train_file, "r") as f:
            train_data = [list(map(int, line.strip().split())) for line in f]
        self.fit(train_data, target_vocab_size)

    def encode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Encode a batch of sequence of integers with merges.
        """
        if not self.merges:
            error_message = "Tokenizer must be fitted or loaded before encoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        if not all(isinstance(units, list) for units in units_list):
            error_message = "Input should be of type list[list[int]]"
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.debug(f"Encoding: {units_list}")

        for i, units in enumerate(units_list):
            while len(units) >= 2:
                counts = self._get_counts([units])
                pair_to_merge = min(
                    counts, key=lambda pair: self.merges.get(pair, float("inf"))
                )
                if pair_to_merge not in self.merges:
                    break
                idx = self.merges[pair_to_merge]
                units = self._merge([units], pair_to_merge, idx)[0]
            units_list[i] = units

        self.logger.info("Finished encoding.")
        self.logger.debug(f"Encoded: {units_list}")

        return units_list

    def decode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Decode a batch of sequence of integers with merges.
        """
        if not self.merges:
            error_message = "Tokenizer must be fitted or loaded before encoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        if not all(isinstance(units, list) for units in units_list):
            error_message = "Input should be of type list[list[int]]"
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.debug(f"Decoding: {units_list}")

        if not self.swapped_merges:
            self.swapped_merges = {v: k for k, v in self.merges.items()}

        for j, units in enumerate(units_list):
            set_units = set(units)
            while set_units & set(self.swapped_merges.keys()):
                decoded_units = []
                i = 0
                while i < len(units):
                    if units[i] in self.swapped_merges:
                        pair = self.swapped_merges[units[i]]
                        decoded_units.extend(pair)
                        i += 1
                    else:
                        decoded_units.append(units[i])
                        i += 1
                units = decoded_units
                set_units = set(units)
            units_list[j] = units

        self.logger.info("Finished decoding.")
        self.logger.debug(f"Decoded: {units_list}")

        return units_list

    def save(self, json_file: str) -> None:
        """
        Save the tokenizer to a file.
        """
        if not self.merges:
            error_message = "Tokenizer must be fitted or loaded before saving."
            self.logger.error(error_message)
            raise ValueError(error_message)

        if not self.swapped_merges:
            self.swapped_merges = {v: k for k, v in self.merges.items()}

        self.logger.debug(f"merges: {self.merges}")
        self.logger.debug(f"swapped_merges: {self.swapped_merges}")

        with open(json_file, "w") as f:
            json.dump(self.swapped_merges, f)

        self.logger.info(f"Tokenizer saved to {json_file}.")

    def load(self, json_file: str) -> None:
        """
        Load the tokenizer from a file.
        """

        with open(json_file, "r") as f:
            self.swapped_merges = json.load(f)

        self.swapped_merges = {int(k): tuple(v) for k, v in self.swapped_merges.items()}
        self.merges = {v: k for k, v in self.swapped_merges.items()}

        self.logger.debug(f"merges: {self.merges}")
        self.logger.debug(f"swapped_merges: {self.swapped_merges}")

        self.logger.info(f"Tokenizer loaded from {json_file}.")
