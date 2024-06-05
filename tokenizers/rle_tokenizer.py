class RLETokenizer:
    """
    Run Length Encoding Tokenizer.
    First max_consecutive_length units (0, ..., max_consecutive_length - 1) are reserved to denote the length of consecutive units. (0 is actually not used.)
    Unit numbers are shifted by max_consecutive_length to avoid conflict with the reserved units.
    """

    def __init__(self):
        self.max_consecutive_length = 100

    def encode(self, units: list[int]):
        """
        Encode a sequence of units.
        """
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
        return encoded

    def decode(self, encoded: list[int]):
        """
        Decode a sequence of encoded units.
        """
        units = []
        for i in range(0, len(encoded), 2):
            consecutive = encoded[i]
            unit = encoded[i + 1]
            units.extend([unit - self.max_consecutive_length] * consecutive)
        return units
