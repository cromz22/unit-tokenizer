import logging

from tokenizers import BaseTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class PackBitsTokenizer(BaseTokenizer):
    """
    PackBits tokenizer that operates on a sequence of units.
    First max_consecutive_length units (0, ..., max_consecutive_length - 1) are reserved to denote the length of consecutive units.
    0 is reserved as the special token to denote that the units after the next unit cannot be compressed.
    Unit numbers are shifted by max_consecutive_length to avoid conflict with the reserved units.
    Example:
        Original units: 0 0 0 0 1 1 2 2 2 2 2 2 3 4 5 6
        Shifted units: 100 100 100 100 101 101 102 102 102 102 102 102 103 104 105 106
        Encoded units: 4 100 2 101 6 102 0 4 103 104 105 106
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.uncompressed_marker = 0
        self.max_consecutive_length = 100

    def _encode(self, units: list[int]) -> list[int]:
        """
        Encode a sequence of units.
        """
        self.logger.debug(f"Encoding: {units}")

        units = [unit + self.max_consecutive_length for unit in units]

        encoded = []
        i = 0
        n = len(units)

        while i < n:
            run_length = 1
            while i + run_length < n and units[i] == units[i + run_length]:
                run_length += 1

            if run_length > 1:
                encoded.append(run_length)
                encoded.append(units[i])
                i += run_length
            else:
                start = i
                while i < n and (i + 1 >= n or units[i] != units[i + 1]):
                    i += 1
                if i > start:
                    encoded.append(self.uncompressed_marker)
                    encoded.append(i - start)
                    encoded.extend(units[start:i])

        return encoded

    def _decode(self, units: list[int]) -> list[int]:
        """
        Decode a sequence of encoded units.
        """
        decoded = []
        i = 0
        n = len(units)

        while i < n:
            if units[i] == self.uncompressed_marker:
                run_length = units[i + 1]
                decoded.extend(units[i + 2 : i + 2 + run_length])
                i += 2 + run_length
            else:
                run_length = units[i]
                decoded.extend([units[i + 1]] * run_length)
                i += 2

        return [unit - self.max_consecutive_length for unit in decoded]
