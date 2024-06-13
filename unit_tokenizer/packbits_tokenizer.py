import logging

from unit_tokenizer import BaseTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class PackBitsTokenizer(BaseTokenizer):
    """
    PackBits tokenizer that operates on a sequence of units.
    First max_run_length units (1, ..., max_run_length) are reserved to denote run length (i.e., the number of consecutive units of the same value).
    0 is reserved as the special token to denote that the units after the next unit cannot be compressed.
    Unit numbers are shifted by `shift` (usually set to max_run_length + 1) to avoid conflict with the reserved units. (Note: This is because the algorithm is targeted for integer encoding alphabets and each integer is meant to have a unique "meaning" in the encoded sequence.)
    If the run length exceeds max_run_length, the sequence will be separated by max_run_length. (e.g., 15 consecutive elements are separated to 10 and 5 when max_run_length=10.)
    """

    def __init__(self, max_run_length=99, shift=100) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.uncompressed_marker = 0
        self.max_run_length = max_run_length
        self.shift = shift
        assert self.max_run_length < self.shift
        if self.max_run_length + 1 != self.shift:
            self.logger.warning(
                f"shift ({self.shift}) should be max_run_length + 1 ({self.max_run_length + 1}) for optimal resource usage."
            )

    def _encode(self, units: list[int]) -> list[int]:
        """
        Encode a sequence of units.
        """
        assert all(unit >= 0 for unit in units)

        units = [unit + self.shift for unit in units]

        encoded = []
        i = 0
        n = len(units)

        while i < n:
            run_length = 1
            while i + run_length < n and units[i] == units[i + run_length]:
                run_length += 1

            if run_length == 1:
                start = i
                while i < n and (i + 1 >= n or units[i] != units[i + 1]):
                    i += 1
                if i > start:
                    encoded.extend(
                        [self.uncompressed_marker, i - start, *units[start:i]]
                    )
            elif run_length > 1 and run_length <= self.max_run_length:
                encoded.extend([run_length, units[i]])
                i += run_length
            else:
                encoded.extend([self.max_run_length, units[i]])
                i += self.max_run_length

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

        return [unit - self.shift for unit in decoded]

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
