import logging

from unit_tokenizer import BaseTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class PackBitsTokenizer(BaseTokenizer):
    """
    PackBits tokenizer that operates on a sequence of units.
    First max_run_length units (0, ..., max_run_length - 1) are reserved to denote run length (number of consecutive units of the same value).
    0 is reserved as the special token to denote that the units after the next unit cannot be compressed.
    Unit numbers are shifted by max_run_length to avoid conflict with the reserved units.
    Example:
        Original units: 0 0 0 0 1 1 2 2 2 2 2 2 3 4 5 6
        Shifted units: 100 100 100 100 101 101 102 102 102 102 102 102 103 104 105 106
        Encoded units: 4 100 2 101 6 102 0 4 103 104 105 106
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.uncompressed_marker = 0
        self.max_run_length = 100

    def _encode(self, units: list[int]) -> list[int]:
        """
        Encode a sequence of units.
        """

        units = [unit + self.max_run_length for unit in units]

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
                start = i
                while i < n and (i + 1 >= n or units[i] != units[i + 1]):
                    i += 1
                if i > start:
                    encoded.extend(
                        [self.uncompressed_marker, i - start, *units[start:i]]
                    )

        return encoded

    def encode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Encode sequences of units.
        """
        if not all(isinstance(units, list) for units in units_list):
            error_message = "Input should be of type list[list[int]]"
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.debug(f"Encoding: {units_list}")

        encoded_list = [self._encode(units) for units in units_list]

        self.logger.info("Finished encoding.")
        self.logger.debug(f"Encoded: {encoded_list}")

        return encoded_list

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

        return [unit - self.max_run_length for unit in decoded]

    def decode(self, units_list: list[list[int]]) -> list[list[int]]:
        """
        Decode sequences of encoded units.
        """
        if not all(isinstance(units, list) for units in units_list):
            error_message = "Input should be of type list[list[int]]"
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.debug(f"Decoding: {units_list}")

        decoded_list = [self._decode(encoded) for encoded in units_list]

        self.logger.info("Finished decoding.")
        self.logger.debug(f"Decoded: {decoded_list}")

        return decoded_list
