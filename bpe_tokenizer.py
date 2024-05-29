import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BaseTokenizer:
    """
    Base tokenizer class.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, train_data: list[int], target_vocab_size: int):
        raise NotImplementedError

    def encode(self, ids: list[int]) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> list[int]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        self.logger.info(f"Saving tokenizer to {path}")

    def load(self, path: str) -> None:
        self.logger.info(f"Loading tokenizer from {path}")



class BPETokenizer(BaseTokenizer):
    """
    Pure BPE tokenizer that operates on a sequence of integers.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.merges = {}
        self.vocab = {}

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
        ids = list(train_data)
        print(ids)

        initial_vocab_size = max(ids)
        num_merges = target_vocab_size - initial_vocab_size
        assert num_merges > 0
        self.logger.info(f"Performing {num_merges} merges")

        merges = {}
        for i in range(num_merges):
            counts = self._get_counts(ids)
            top_pair = max(counts, key=counts.get)
            new_idx = initial_vocab_size + i
            ids = self._merge(ids, top_pair, new_idx)
            merges[top_pair] = new_idx
            self.logger.info(f"Merge {i + 1}/{num_merges}: {top_pair} -> {new_idx}; IDs: {ids}")

        self.logger.info(f"original length: {len(train_data)} -> bpe length: {len(ids)}")
        self.logger.info(f"compression rate: {len(ids) / len(train_data):.2f}")

    def encode(self, ids: list[int]) -> list[int]:
        pass

    def decode(self, ids: list[int]) -> list[int]:
        pass
