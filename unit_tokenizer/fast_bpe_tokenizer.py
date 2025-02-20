import json
import logging
import heapq
from collections import defaultdict
from unit_tokenizer import BaseTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Node:
    __slots__ = ("token", "prev", "next", "active")

    def __init__(self, token: int):
        self.token = token
        self.prev = None
        self.next = None
        self.active = True


class FastBPETokenizer(BaseTokenizer):
    """
    Fast BPE Tokenizer that operates on batches of integer sequences using local updates.
    This version leverages a priority queue to efficiently choose the most frequent pair.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.merges: dict[tuple[int, int], int] = {}
        self.merge_order: list[tuple[tuple[int, int], int]] = []
        self.swapped_merges: dict[int, tuple[int, int]] = {}

    @staticmethod
    def _build_linked_list(seq: list[int]) -> Node:
        head = Node(seq[0])
        current = head
        for token in seq[1:]:
            new_node = Node(token)
            new_node.prev = current
            current.next = new_node
            current = new_node
        return head

    @staticmethod
    def _linked_list_to_list(head: Node) -> list[int]:
        result = []
        node = head
        while node:
            if node.active:
                result.append(node.token)
            node = node.next
        return result

    def fit(self, train_data: list[list[int]], target_vocab_size: int) -> None:
        """
        Learn merge rules from training data until the target vocabulary size is reached.
        This version uses a priority queue to extract the best pair in O(log n) time.
        """
        if not train_data or not any(train_data):
            error_message = "Training data is empty."
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Determine initial vocabulary.
        initial_vocab = {token for seq in train_data for token in seq}
        initial_vocab_size = len(initial_vocab)
        if target_vocab_size <= initial_vocab_size:
            error_message = (
                f"Target vocab size ({target_vocab_size}) must be greater than "
                f"the initial vocab size ({initial_vocab_size})."
            )
            self.logger.error(error_message)
            raise ValueError(error_message)

        num_merges = target_vocab_size - initial_vocab_size
        self.logger.info(f"Fitting tokenizer with {num_merges} merges.")

        # Build linked lists for each sequence.
        linked_sequences = [self._build_linked_list(seq) for seq in train_data]

        # Map each adjacent pair to the set of left nodes.
        pairs: dict[tuple[int, int], set[Node]] = defaultdict(set)
        for head in linked_sequences:
            node = head
            while node and node.next:
                if node.active and node.next.active:
                    pairs[(node.token, node.next.token)].add(node)
                node = node.next

        merge_rules = []
        next_new_token = max(initial_vocab) + 1

        # Build a count dictionary and a max-heap (using negative counts).
        pairs_count = {pair: len(nodes) for pair, nodes in pairs.items()}
        priority_queue = []
        for pair, count in pairs_count.items():
            heapq.heappush(priority_queue, (-count, pair))

        for merge_index in range(num_merges):
            best_pair = None
            best_count = 0

            # Extract the pair with the highest frequency.
            while priority_queue:
                neg_count, pair = heapq.heappop(priority_queue)
                count = -neg_count
                # Check if this count is up-to-date.
                if pairs_count.get(pair, 0) == count:
                    best_pair = pair
                    best_count = count
                    break

            if best_pair is None or best_count == 0:
                self.logger.warning("No more valid pairs to merge.")
                break

            a, b = best_pair
            new_token = next_new_token
            next_new_token += 1
            merge_rules.append((best_pair, new_token))
            self.merges[best_pair] = new_token
            self.logger.info(f"Merge {merge_index+1}/{num_merges}: {best_pair} -> {new_token}")

            update_count_pairs = set()

            # Process all valid occurrences of best_pair.
            for node in list(pairs[best_pair]):
                if not (node.active and node.next and node.next.active and (node.token, node.next.token) == best_pair):
                    continue
                node.token = new_token
                removed = node.next
                node.next = removed.next
                if removed.next:
                    removed.next.prev = node

                # Update neighboring pairs.
                if node.prev:
                    old_pair = (node.prev.token, a)
                    if node.prev in pairs[old_pair]:
                        pairs[old_pair].discard(node.prev)
                        update_count_pairs.add(old_pair)
                    new_pair = (node.prev.token, node.token)
                    pairs[new_pair].add(node.prev)
                    update_count_pairs.add(new_pair)
                if node.next:
                    new_pair = (node.token, node.next.token)
                    pairs[new_pair].add(node)
                    update_count_pairs.add(new_pair)
                    old_pair = (b, node.next.token)
                    if removed in pairs[old_pair]:
                        pairs[old_pair].discard(removed)
                        update_count_pairs.add(old_pair)

            # Refresh counts in the priority queue for affected pairs.
            for pair in update_count_pairs:
                pairs_count[pair] = len(pairs[pair])
                heapq.heappush(priority_queue, (-pairs_count[pair], pair))
            pairs_count[best_pair] = 0

        # Store the merge order.
        self.merge_order = merge_rules
        self.logger.info("Finished fitting tokenizer.")

    def fit_from_file(self, train_file: str, target_vocab_size: int) -> None:
        """
        Fit the tokenizer from a file. Each line should contain a sequence of integers
        separated by spaces.
        """
        with open(train_file, "r") as f:
            train_data = [list(map(int, line.strip().split())) for line in f]
        self.fit(train_data, target_vocab_size)

    def encode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Encode a batch of integer sequences by applying each learned merge rule in order.
        """
        if not self.merge_order:
            error_message = "Tokenizer must be fitted or loaded before encoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Work on a copy of the input.
        encoded = [seq[:] for seq in units_list]
        for pair, new_token in self.merge_order:
            for i in range(len(encoded)):
                seq = encoded[i]
                new_seq = []
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and (seq[j], seq[j + 1]) == pair:
                        new_seq.append(new_token)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1
                encoded[i] = new_seq
        self.logger.info("Finished encoding.")
        return encoded

    def decode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Decode a batch of integer sequences by recursively expanding merged tokens.
        """
        if not self.merges:
            error_message = "Tokenizer must be fitted or loaded before decoding."
            self.logger.error(error_message)
            raise ValueError(error_message)

        if not self.swapped_merges:
            self.swapped_merges = {v: k for k, v in self.merges.items()}

        memo: dict[int, list[int]] = {}

        def recursive_decode(token: int) -> list[int]:
            if token in memo:
                return memo[token]
            if token in self.swapped_merges:
                a, b = self.swapped_merges[token]
                result = recursive_decode(a) + recursive_decode(b)
            else:
                result = [token]
            memo[token] = result
            return result

        decoded = []
        for seq in units_list:
            new_seq: list[int] = []
            for token in seq:
                new_seq.extend(recursive_decode(token))
            decoded.append(new_seq)
        self.logger.info("Finished decoding.")
        return decoded

    def save(self, json_file: str) -> None:
        """
        Save the learned tokenizer to a JSON file.
        We now store the ordered merge rules so that encoding works after loading.
        """
        if not self.merge_order:
            error_message = "Tokenizer must be fitted or loaded before saving."
            self.logger.error(error_message)
            raise ValueError(error_message)
        data = {
            "merge_order": [
                [pair[0], pair[1], new_token] for (pair, new_token) in self.merge_order
            ]
        }
        with open(json_file, "w") as f:
            json.dump(data, f)
        self.logger.info(f"Tokenizer saved to {json_file}.")

    def load(self, json_file: str) -> None:
        """
        Load the tokenizer from a JSON file.
        """
        with open(json_file, "r") as f:
            data = json.load(f)
        self.merge_order = []
        self.merges = {}
        for item in data.get("merge_order", []):
            a, b, new_token = item
            pair = (a, b)
            self.merge_order.append((pair, new_token))
            self.merges[pair] = new_token
        self.swapped_merges = {v: k for k, v in self.merges.items()}
        self.logger.info(f"Tokenizer loaded from {json_file}.")

