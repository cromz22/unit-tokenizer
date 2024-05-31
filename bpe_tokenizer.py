import logging
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BPETokenizer:
    """
    Pure BPE tokenizer that operates on a sequence of integers.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.merges = {}
        self.swapped_merges = {}

    def _get_counts(self, ids_seq: list[list[int]]) -> dict[tuple[int, int], int]:
        """
        Count the number of occurrences for each pair of ids within each inner list.
        """
        counts = {}
        for ids in ids_seq:
            for pair in zip(ids[:-1], ids[1:]):
                if pair in counts:
                    counts[pair] += 1
                else:
                    counts[pair] = 1
        return counts

    def _merge(self, ids_seq: list[list[int]], pair: tuple[int, int], idx: int) -> list[list[int]]:
        """
        Replace all occurrences of `pair` in `ids_seq` with `idx`.
        """
        new_ids_seq = []
        for ids in ids_seq:
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_ids_seq.append(new_ids)
        return new_ids_seq

    def fit(self, train_data: list[list[int]], target_vocab_size: int):
        """
        Fit the tokenizer on `train_data`.
        """
        if not train_data or not any(train_data):
            error_message = "Training data is empty."
            self.logger.error(error_message)
            raise ValueError(error_message)

        ids_seq = train_data
        set_ids_seq = [set(ids) for ids in ids_seq]
        set_ids = set.union(*set_ids_seq)
        initial_vocab_size = len(set_ids)
        max_id = max(set_ids)

        if target_vocab_size <= initial_vocab_size:
            error_message = f"Target vocab size ({target_vocab_size}) must be greater than the initial vocab size ({initial_vocab_size})."
            self.logger.error(error_message)
            raise ValueError(error_message)

        num_merges = target_vocab_size - initial_vocab_size
        self.logger.info(f"Fitting tokenizer with {num_merges} merges.")
        self.logger.debug(f"Initial IDs: {ids_seq}")

        for i in range(num_merges):
            counts = self._get_counts(ids_seq)
            if not counts:
                self.logger.warning("No more pairs to merge.")
                break
            top_pair = max(counts, key=counts.get)
            new_idx = max_id + 1
            ids_seq = self._merge(ids_seq, top_pair, new_idx)
            self.merges[top_pair] = new_idx
            self.logger.info(f"Merge {i + 1}/{num_merges}: {top_pair} -> {new_idx}")
            self.logger.debug(f"IDs: {ids_seq}")

            max_id = new_idx


    def encode(self, ids_seq: list[list[int]]) -> list[list[int]]:
        """
        Encode a batch of sequence of integers with merges.
        """
        if not self.merges:
            error_message = "Tokenizer must be fitted or loaded before encoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.debug(f"Encoding: {ids_seq}")

        for i, ids in enumerate(ids_seq):
            while len(ids) >= 2:
                counts = self._get_counts([ids])
                pair_to_merge = min(counts, key=lambda pair: self.merges.get(pair, float('inf')))
                if pair_to_merge not in self.merges:
                    break
                idx = self.merges[pair_to_merge]
                ids = self._merge([ids], pair_to_merge, idx)[0]
            ids_seq[i] = ids

        self.logger.info("Finished encoding.")
        self.logger.debug(f"Encoded: {ids_seq}")

        return ids_seq

    def decode(self, ids_seq: list[list[int]]) -> list[list[int]]:
        """
        Decode a batch of sequence of integers with merges.
        """
        if not self.merges:
            error_message = "Tokenizer must be fitted or loaded before encoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.debug(f"Decoding: {ids_seq}")

        if not self.swapped_merges:
            self.swapped_merges = {v: k for k, v in self.merges.items()}
        
        for j, ids in enumerate(ids_seq):
            ids_set = set(ids)
            while ids_set & set(self.swapped_merges.keys()):
                decoded_ids = []
                i = 0
                while i < len(ids):
                    if ids[i] in self.swapped_merges:
                        pair = self.swapped_merges[ids[i]]
                        decoded_ids.extend(pair)
                        i += 1
                    else:
                        decoded_ids.append(ids[i])
                        i += 1
                ids = decoded_ids
                ids_set = set(ids)
            ids_seq[j] = ids

        self.logger.info("Finished decoding.")
        self.logger.debug(f"Decoded: {ids_seq}")

        return ids_seq

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

        with open(json_file, 'w') as f:
            json.dump(self.swapped_merges, f)

        self.logger.info(f"Tokenizer saved to {json_file}.")

    def load(self, json_file: str) -> None:
        """
        Load the tokenizer from a file.
        """

        with open(json_file, 'r') as f:
            self.swapped_merges = json.load(f)

        self.swapped_merges = {int(k): tuple(v) for k, v in self.swapped_merges.items()}
        self.merges = {v: k for k, v in self.swapped_merges.items()}

        self.logger.debug(f"merges: {self.merges}")
        self.logger.debug(f"swapped_merges: {self.swapped_merges}")

        self.logger.info(f"Tokenizer loaded from {json_file}.")
