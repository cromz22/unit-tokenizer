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
        self.merge_rules: list[tuple[tuple[int, int], int]] = []

    @property
    def reverse_merge_mapping(self) -> dict[int, tuple[int, int]]:
        """Compute reverse mapping from merged token to pair."""
        return {new_token: pair for pair, new_token in self.merge_rules}

    def _get_counts(self, units_list: list[list[int]]) -> dict[tuple[int, int], int]:
        """
        Count the number of occurrences for each pair of units within each inner list.
        """
        counts = {}
        for units in units_list:
            for pair in zip(units[:-1], units[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, units_list: list[list[int]], pair: tuple[int, int], idx: int) -> list[list[int]]:
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
            error_message = (
                f"Target vocab size ({target_vocab_size}) must be greater than the initial vocab size ({initial_vocab_size})."
            )
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
            top_pair, _count = max(counts.items(), key=lambda x: (x[1], -x[0][0], -x[0][1]))
            new_idx = max_idx + 1
            units_list = self._merge(units_list, top_pair, new_idx)
            self.merge_rules.append((top_pair, new_idx))
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
        Encode a batch of sequences of integers by sequentially applying the learned merge rules in order.
        """
        if not self.merge_rules:
            error_message = "Tokenizer must be fitted or loaded before encoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.debug(f"Encoding: {units_list}")
        encoded_units_list = []
        for units in units_list:
            # Apply each merge rule in the order they were learned.
            for pair, new_token in self.merge_rules:
                units = self._merge([units], pair, new_token)[0]
            encoded_units_list.append(units)
        self.logger.info("Finished encoding.")
        self.logger.debug(f"Encoded: {encoded_units_list}")
        return encoded_units_list

    def decode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Decode a batch of sequences of integers by iteratively replacing merged tokens with their original pairs.
        """
        if not self.merge_rules:
            error_message = "Tokenizer must be fitted or loaded before decoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        reverse_mapping = self.reverse_merge_mapping
        self.logger.debug(f"Decoding: {units_list}")

        decoded_units_list = []
        for units in units_list:
            # Repeatedly replace tokens that were merged.
            while any(token in reverse_mapping for token in units):
                decoded_units = []
                i = 0
                while i < len(units):
                    if units[i] in reverse_mapping:
                        decoded_units.extend(reverse_mapping[units[i]])
                        i += 1
                    else:
                        decoded_units.append(units[i])
                        i += 1
                units = decoded_units
            decoded_units_list.append(units)
        self.logger.info("Finished decoding.")
        self.logger.debug(f"Decoded: {decoded_units_list}")
        return decoded_units_list

    def save(self, json_file: str) -> None:
        """
        Save the tokenizer to a file in a unified format.
        The saved JSON contains a key "merge_rules" whose value is a list of triples [a, b, new_token].
        """
        if not self.merge_rules:
            error_message = "Tokenizer must be fitted or loaded before saving."
            self.logger.error(error_message)
            raise ValueError(error_message)

        data = {
            "merge_rules": [[a, b, new_token] for ((a, b), new_token) in self.merge_rules]
        }
        with open(json_file, "w") as f:
            json.dump(data, f)
        self.logger.info(f"Tokenizer saved to {json_file}.")

    def load(self, json_file: str) -> None:
        """
        Load the tokenizer from a file in the unified format.
        Reconstructs the merge rules.
        """
        with open(json_file, "r") as f:
            data = json.load(f)
        merge_rules_data = data.get("merge_rules")
        if merge_rules_data is None:
            error_message = "Invalid file format."
            self.logger.error(error_message)
            raise ValueError(error_message)
        self.merge_rules = [((item[0], item[1]), item[2]) for item in merge_rules_data]
        self.logger.debug(f"merge_rules: {self.merge_rules}")
        self.logger.info(f"Tokenizer loaded from {json_file}.")
