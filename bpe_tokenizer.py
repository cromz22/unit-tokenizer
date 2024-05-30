import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BPETokenizer:
    """
    Pure BPE tokenizer that operates on a sequence of integers.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.merges = {}

    def _get_counts(self, ids: list[int]) -> dict[tuple[int, int], int]:
        """
        Count the number of occurrences for each pair of ids.
        """
        counts = {}
        for pair in zip(ids[:-1], ids[1:]):
            if pair in counts:
                counts[pair] += 1
            else:
                counts[pair] = 1
        return counts

    def _merge(self, ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """
        Replace all occurrences of `pair` in `ids` with `idx`.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def fit(self, train_data: list[int], target_vocab_size: int):
        """
        Fit the tokenizer on the training data.
        """
        if not train_data:
            error_message = "Training data is empty."
            self.logger.error(error_message)
            raise ValueError(error_message)

        ids = list(train_data)
        initial_vocab_size = len(set(ids))

        if target_vocab_size <= initial_vocab_size:
            error_message = f"Target vocab size ({target_vocab_size}) must be greater than the initial vocab size ({initial_vocab_size})."
            self.logger.error(error_message)
            raise ValueError(error_message)

        num_merges = target_vocab_size - initial_vocab_size
        self.logger.info(f"Performing {num_merges} merges. IDs: {ids}")

        for i in range(num_merges):
            counts = self._get_counts(ids)
            if not counts:
                self.logger.warning("No more pairs to merge.")
                break
            top_pair = max(counts, key=counts.get)
            new_idx = initial_vocab_size + i
            ids = self._merge(ids, top_pair, new_idx)
            self.merges[top_pair] = new_idx
            self.logger.info(f"Merge {i + 1}/{num_merges}: {top_pair} -> {new_idx}; IDs: {ids}")

        self.logger.info(f"original length: {len(train_data)} -> bpe length: {len(ids)} (compression rate: {len(ids) / len(train_data):.2f})")

    def encode(self, ids: list[int]) -> list[int]:
        """
        Encode a sequence of integers with merges.
        """
        if not self.merges:
            error_message = "Tokenizer must be fitted or loaded before encoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.info(f"Encoding: {ids}")

        while len(ids) >= 2:
            counts = self._get_counts(ids)
            pair_to_merge = min(counts, key=lambda pair: self.merges.get(pair, float('inf')))
            if pair_to_merge not in self.merges:
                break
            idx = self.merges[pair_to_merge]
            ids = self._merge(ids, pair_to_merge, idx)

        self.logger.info(f"Encoded: {ids}")

        return ids

    def decode(self, ids: list[int]) -> list[int]:
        """
        Decode a sequence of integers with merges.
        """
        if not self.merges:
            error_message = "Tokenizer must be fitted or loaded before encoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.info(f"Decoding: {ids}")

        reverse_merges = {v: k for k, v in self.merges.items()}
        
        ids_set = set(ids)
        while ids_set & set(reverse_merges.keys()):
            decoded_ids = []
            i = 0
            while i < len(ids):
                if ids[i] in reverse_merges:
                    pair = reverse_merges[ids[i]]
                    decoded_ids.extend(pair)
                    i += 1
                else:
                    decoded_ids.append(ids[i])
                    i += 1
            ids = decoded_ids
            ids_set = set(ids)

        self.logger.info(f"Decoded: {ids}")

        return ids

    def save(self, json_file: str) -> None:
        """
        Save the tokenizer to a file.
        """
        with open(json_file, 'w') as f:
            f.write(str(self.merges))

        self.logger.info(f"Tokenizer saved to {json_file}.")

    def load(self, json_file: str) -> None:
        """
        Load the tokenizer from a file.
        """
        with open(json_file, 'r') as f:
            self.merges = eval(f.read())

        self.logger.info(f"Tokenizer loaded from {json_file}.")
