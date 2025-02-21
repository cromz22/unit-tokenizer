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
        for unit in seq[1:]:
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
        """
        Learn merge rules from training data until the target vocabulary size is reached.
        This version uses a priority queue to extract the best pair in O(log n) time.
        """
        if not units_list or not any(units_list):
            error_message = "Training data is empty."
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Determine initial vocabulary.
        initial_vocab = {unit for units in units_list for unit in units}
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
        linked_units_list = [self._build_linked_list(units) for units in units_list]

        # Map each adjacent pair to the set of left nodes.
        pairs: dict[tuple[int, int], set[Node]] = defaultdict(set)
        for head in linked_units_list:
            node = head
            while node and node.next:
                if node.active and node.next.active:
                    pairs[(node.unit, node.next.unit)].add(node)
                node = node.next

        merge_order = []
        next_new_unit = max(initial_vocab) + 1

        # Build a count dictionary and a max-heap (using negative counts).
        pairs_count = {pair: len(nodes) for pair, nodes in pairs.items()}
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
                # Check if this count is up-to-date.
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
            merge_order.append((most_frequent_pair, new_unit))
            self.merges[most_frequent_pair] = new_unit
            self.logger.info(f"Merge {i+1}/{num_merges}: {most_frequent_pair} -> {new_unit}")

            update_count_pairs = set()

            # Process all valid occurrences of best_pair.
            for node in list(pairs[most_frequent_pair]):
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
                    if node.prev in pairs[old_pair]:
                        pairs[old_pair].discard(node.prev)
                        update_count_pairs.add(old_pair)
                    new_pair = (node.prev.unit, node.unit)
                    pairs[new_pair].add(node.prev)
                    update_count_pairs.add(new_pair)
                if node.next:
                    new_pair = (node.unit, node.next.unit)
                    pairs[new_pair].add(node)
                    update_count_pairs.add(new_pair)
                    old_pair = (b, node.next.unit)
                    if removed in pairs[old_pair]:
                        pairs[old_pair].discard(removed)
                        update_count_pairs.add(old_pair)

            # Refresh counts in the priority queue for affected pairs.
            for pair in update_count_pairs:
                pairs_count[pair] = len(pairs[pair])
                heapq.heappush(priority_queue, (-pairs_count[pair], pair))

            pairs_count[most_frequent_pair] = 0

        # Store the merge order.
        self.merge_order = merge_order
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
        For each sequence, apply the learned merge rules in order.
        """
        encoded_sequences = []
        for sequence in units_list:
            units = sequence.copy()
            for (a, b), new_unit in self.merge_order:
                i = 0
                new_units = []
                # Scan units left-to-right merging every occurrence of (a, b)
                while i < len(units):
                    if i < len(units) - 1 and units[i] == a and units[i + 1] == b:
                        new_units.append(new_unit)
                        i += 2
                    else:
                        new_units.append(units[i])
                        i += 1
                units = new_units
            encoded_sequences.append(units)
        return encoded_sequences

    def decode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Recursively expand merged units using the stored swapped_merges.
        """
        if not self.swapped_merges:
            self.swapped_merges = {v: k for k, v in self.merges.items()}

        def recursive_decode(unit: int) -> list[int]:
            if unit in self.swapped_merges:
                a, b = self.swapped_merges[unit]
                return recursive_decode(a) + recursive_decode(b)
            else:
                return [unit]

        decoded_sequences = []
        for sequence in units_list:
            decoded_sequence = []
            for unit in sequence:
                decoded_sequence.extend(recursive_decode(unit))
            decoded_sequences.append(decoded_sequence)
        return decoded_sequences


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
                [pair[0], pair[1], new_units] for (pair, new_units) in self.merge_order
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
            a, b, new_unit = item
            pair = (a, b)
            self.merge_order.append((pair, new_unit))
            self.merges[pair] = new_unit
        self.swapped_merges = {v: k for k, v in self.merges.items()}
        self.logger.info(f"Tokenizer loaded from {json_file}.")

