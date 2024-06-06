import logging

from unit_tokenizer import BaseTokenizer

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
        i = 0
        n = len(units)

        while i < n:
            run_length = 1
            while i + run_length < n and units[i] == units[i + run_length]:
                run_length += 1

            if run_length > 1:
                encoded.extend([run_length, units[i]])
                i += run_length
            else:
                encoded.extend([1, units[i]])
                i += 1

        self.logger.info("Finished encoding.")
        self.logger.debug(f"Encoded: {encoded}")

        return encoded

    def encode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Encode sequences of units.
        """
        if not all(isinstance(units, list) for units in units_list):
            error_message = "Input should be of type list[list[int]]"
            self.logger.error(error_message)
            raise ValueError(error_message)

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

    def decode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Decode sequences of encoded units.
        """
        if not all(isinstance(units, list) for units in units_list):
            error_message = "Input should be of type list[list[int]]"
            self.logger.error(error_message)
            raise ValueError(error_message)

        return [self._decode(encoded) for encoded in units_list]
