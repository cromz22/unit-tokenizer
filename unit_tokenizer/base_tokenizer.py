class BaseTokenizer:
    """
    Base class for all tokenizers.
    """

    def __init__(self):
        pass

    def encode(self, units_list: list[list[int]]) -> list[list[int]]:
        raise NotImplementedError

    def decode(self, units_list: list[list[int]]) -> list[list[int]]:
        raise NotImplementedError

    def encode_from_file(self, input_file: str, output_file: str) -> None:
        """
        Encode from file input and save the encoded sequences to an output file.
        `input_file` should contain a sequence of integers separated by spaces per line.
        `output_file` will contain the encoded sequences.
        """
        with open(input_file, "r") as f:
            units_list = [list(map(int, line.strip().split())) for line in f]
        encoded_units_list = self.encode(units_list)
        with open(output_file, "w") as f:
            for units in encoded_units_list:
                f.write(" ".join(map(str, units)) + "\n")

    def decode_from_file(self, input_file: str, output_file: str) -> None:
        """
        Decode from file input and save the decoded sequences to an output file.
        `input_file` should contain a sequence of integers separated by spaces per line.
        `output_file` will contain the decoded sequences.
        """
        with open(input_file, "r") as f:
            units_list = [list(map(int, line.strip().split())) for line in f]
        decoded_units_list = self.decode(units_list)
        with open(output_file, "w") as f:
            for units in decoded_units_list:
                f.write(" ".join(map(str, units)) + "\n")
