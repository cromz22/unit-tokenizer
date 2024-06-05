import logging

from tokenizers import BaseTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class RLETokenizer(BaseTokenizer):
    """
    Run Length Encoding Tokenizer that operates on a sequence of units.
    First max_consecutive_length units (0, ..., max_consecutive_length - 1) are reserved to denote the length of consecutive units. (0 is actually not used.)
    Unit numbers are shifted by max_consecutive_length to avoid conflict with the reserved units.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_consecutive_length = 100

    def _encode(self, units: list[int]) -> list[int]:
        """
        Encode a sequence of units.
        """
        self.logger.debug(f"Encoding: {units}")

        units = [unit + self.max_consecutive_length for unit in units]

        encoded = []
        consecutive = 1
        for i in range(1, len(units)):
            if units[i] == units[i - 1]:
                consecutive += 1
            else:
                encoded.append(consecutive)
                encoded.append(units[i - 1])
                consecutive = 1
        encoded.append(consecutive)
        encoded.append(units[-1])

        self.logger.info("Finished encoding.")
        self.logger.debug(f"Encoded: {encoded}")

        return encoded

    def encode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Encode sequences of units.
        """
        return [self._encode(units) for units in units_list]

    def _decode(self, encoded: list[int]) -> list[int]:
        """
        Decode a sequence of encoded units.
        """
        units = []
        for i in range(0, len(encoded), 2):
            consecutive = encoded[i]
            unit = encoded[i + 1]
            units.extend([unit - self.max_consecutive_length] * consecutive)
        return units

    def decode(self, encoded_list: list[list[int]]) -> list[list[int]]:
        """
        Decode sequences of encoded units.
        """
        return [self._decode(encoded) for encoded in encoded_list]
