import json
import logging
import heapq
from collections import defaultdict
from unit_tokenizer import BaseTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Node:
    __slots__ = ("unit", "prev", "next", "active")
    def __init__(self, unit: int):
        self.unit = unit
        self.prev = None
        self.next = None
        self.active = True


class FastBPETokenizer(BaseTokenizer):
    """
    Fast BPE Tokenizer for integer sequences.
    A priority queue is used for fitting.
    A trie used for encoding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.merge_rules: list[tuple[tuple[int, int], int]] = []
        self.decoded_merges: dict[int, list[int]] = {}  # Cache: maps each token (base or merged) to its fully decoded sequence.
        self.trie: dict = {}  # Trie for fast matching in encoding.
        self.initial_vocab_max: int = 0  # Maximum initial token value (used to distinguish initial tokens from merged tokens).

    @staticmethod
    def _build_linked_list(units: list[int]) -> Node:
        head = Node(units[0])
        current = head
        for unit in units[1:]:
            new_node = Node(unit)
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
                result.append(node.unit)
            node = node.next
        return result

    def fit(self, units_list: list[list[int]], target_vocab_size: int) -> None:
        if not units_list or not any(units_list):
            error_message = "Training data is empty."
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Determine the initial vocabulary.
        initial_vocab = {unit for units in units_list for unit in units}
        self.initial_vocab_max = max(initial_vocab)
        initial_vocab_size = len(initial_vocab)
        if target_vocab_size <= initial_vocab_size:
            error_message = (
                f"Target vocab size ({target_vocab_size}) must be greater than "
                f"the initial vocab size ({initial_vocab_size})."
            )
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Initialize cache with base tokens.
        for token in initial_vocab:
            self.decoded_merges[token] = [token]

        num_merges = target_vocab_size - initial_vocab_size
        self.logger.info(f"Fitting tokenizer with {num_merges} merges.")

        # Build linked lists for each sequence.
        linked_units_list: list[Node] = [self._build_linked_list(units) for units in units_list]

        # Map each adjacent pair to the set of left nodes.
        pairs_positions: dict[tuple[int, int], set[Node]] = defaultdict(set)
        for head in linked_units_list:
            node = head
            while node and node.next:
                if node.active and node.next.active:
                    pairs_positions[(node.unit, node.next.unit)].add(node)
                node = node.next

        merge_rules = []
        next_new_unit = self.initial_vocab_max + 1

        # Build a count dictionary and a max-heap (using negative counts).
        pairs_count = {pair: len(nodes) for pair, nodes in pairs_positions.items()}
        priority_queue = []
        for pair, count in pairs_count.items():
            heapq.heappush(priority_queue, (-count, pair))

        for i in range(num_merges):
            most_frequent_pair = None
            most_frequent_count = 0
            # Extract the pair with the highest frequency.
            while priority_queue:
                neg_count, pair = heapq.heappop(priority_queue)
                count = -neg_count
                if pairs_count.get(pair, 0) == count:
                    most_frequent_pair = pair
                    most_frequent_count = count
                    break

            if most_frequent_pair is None or most_frequent_count == 0:
                self.logger.warning("No more valid pairs to merge.")
                break

            a, b = most_frequent_pair
            new_unit = next_new_unit
            next_new_unit += 1
            merge_rules.append((most_frequent_pair, new_unit))
            self.decoded_merges[new_unit] = self.decoded_merges[a] + self.decoded_merges[b]
            self.logger.info(f"Merge {i+1}/{num_merges}: {most_frequent_pair} -> {new_unit}")

            update_count_pairs = set()
            # Process all valid occurrences of the most frequent pair.
            for node in list(pairs_positions[most_frequent_pair]):
                if not (node.active and node.next and node.next.active and (node.unit, node.next.unit) == most_frequent_pair):
                    continue
                node.unit = new_unit
                removed = node.next
                node.next = removed.next
                if removed.next:
                    removed.next.prev = node

                # Update neighboring pairs.
                if node.prev:
                    old_pair = (node.prev.unit, a)
                    if node.prev in pairs_positions[old_pair]:
                        pairs_positions[old_pair].discard(node.prev)
                        update_count_pairs.add(old_pair)
                    new_pair = (node.prev.unit, node.unit)
                    pairs_positions[new_pair].add(node.prev)
                    update_count_pairs.add(new_pair)
                if node.next:
                    new_pair = (node.unit, node.next.unit)
                    pairs_positions[new_pair].add(node)
                    update_count_pairs.add(new_pair)
                    old_pair = (b, node.next.unit)
                    if removed in pairs_positions[old_pair]:
                        pairs_positions[old_pair].discard(removed)
                        update_count_pairs.add(old_pair)

            # Refresh counts for affected pairs.
            for pair in update_count_pairs:
                pairs_count[pair] = len(pairs_positions[pair])
                heapq.heappush(priority_queue, (-pairs_count[pair], pair))
            pairs_count[most_frequent_pair] = 0

        self.merge_rules = merge_rules
        self.logger.info("Finished fitting tokenizer.")
        self._build_trie()

    def fit_from_file(self, train_file: str, target_vocab_size: int) -> None:
        with open(train_file, "r") as f:
            units_list = [list(map(int, line.strip().split())) for line in f]
        self.fit(units_list, target_vocab_size)

    def _build_trie(self) -> None:
        """
        Build a trie from cached final sequences for all merged tokens.
        Only tokens greater than initial_vocab_max (i.e. non-base tokens) are added.
        """
        self.trie = {}
        for token, seq in self.decoded_merges.items():
            if token <= self.initial_vocab_max:
                continue
            node = self.trie
            for t in seq:
                if t not in node:
                    node[t] = {}
                node = node[t]
            node["token"] = token

    def encode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Encode sequences using a greedy longest-match search in the trie.
        This replaces iterative scanning over the merge order.
        """
        encoded_units_list = []
        for units in units_list:
            i = 0
            encoded_units = []
            while i < len(units):
                node = self.trie
                match = None
                j = i
                # Greedily match the longest sequence.
                while j < len(units) and units[j] in node:
                    node = node[units[j]]
                    if "token" in node:
                        match = (node["token"], j)
                    j += 1
                if match is not None:
                    token, j_match = match
                    encoded_units.append(token)
                    i = j_match + 1
                else:
                    encoded_units.append(units[i])
                    i += 1
            encoded_units_list.append(encoded_units)
        return encoded_units_list

    def decode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Decode by replacing each token with its fully cached final sequence.
        """
        decoded_units_list = []
        for units in units_list:
            decoded_units = []
            for unit in units:
                decoded_units.extend(self.decoded_merges.get(unit, [unit]))
            decoded_units_list.append(decoded_units)
        return decoded_units_list

    def save(self, json_file: str) -> None:
        """
        Save the merge order (and implicitly the cached sequences) to a JSON file.
        """
        if not self.merge_rules:
            error_message = "Tokenizer must be fitted or loaded before saving."
            self.logger.error(error_message)
            raise ValueError(error_message)
        data = {
            "merge_rules": [
                [pair[0], pair[1], new_unit] for (pair, new_unit) in self.merge_rules
            ]
        }
        with open(json_file, "w") as f:
            json.dump(data, f)
        self.logger.info(f"Tokenizer saved to {json_file}.")

    def load(self, json_file: str) -> None:
        """
        Load the merge rules from a JSON file and rebuild caches.
        """
        with open(json_file, "r") as f:
            data = json.load(f)
        self.merge_rules = []
        for item in data.get("merge_rules", []):
            a, b, new_unit = item
            pair = (a, b)
            self.merge_rules.append((pair, new_unit))

        self.decoded_merges = {}
        for pair, new_unit in self.merge_rules:
            a, b = pair
            if a not in self.decoded_merges:
                self.decoded_merges[a] = [a]
            if b not in self.decoded_merges:
                self.decoded_merges[b] = [b]
            self.decoded_merges[new_unit] = self.decoded_merges[a] + self.decoded_merges[b]
        self._build_trie()
        self.logger.info(f"Tokenizer loaded from {json_file}.")
