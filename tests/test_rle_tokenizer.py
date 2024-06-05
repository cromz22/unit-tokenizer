import os

from tokenizers import RLETokenizer


def test_encode():
    tokenizer = RLETokenizer()
    encoded = tokenizer.encode([[0, 1, 2, 2, 3, 3, 3]])
    assert encoded[0] == [1, 100, 1, 101, 2, 102, 3, 103]


def test_decode():
    tokenizer = RLETokenizer()
    decoded = tokenizer.decode([[1, 100, 1, 101, 2, 102, 3, 103]])
    assert decoded[0] == [0, 1, 2, 2, 3, 3, 3]


def test_batch_encode():
    tokenizer = RLETokenizer()
    encoded = tokenizer.encode([[0, 1, 2, 2, 3, 3, 3], [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]])
    assert encoded == [
        [1, 100, 1, 101, 2, 102, 3, 103],
        [1, 100, 2, 101, 3, 102, 4, 103],
    ]


def test_batch_decode():
    tokenizer = RLETokenizer()
    decoded = tokenizer.decode(
        [[1, 100, 1, 101, 2, 102, 3, 103], [1, 100, 2, 101, 3, 102, 4, 103]]
    )
    assert decoded == [[0, 1, 2, 2, 3, 3, 3], [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]]


def test_encode_from_file():
    input_file = "test_encode_from_file_input.txt"
    output_file = "test_encode_from_file_output.txt"

    with open(input_file, "w") as f:
        f.write("0 1 2 2 3 3 3\n0 1 1 2 2 2 3 3 3 3\n")

    tokenizer = RLETokenizer()
    encoded = tokenizer.encode_from_file(input_file, output_file)

    with open(output_file, "r") as f:
        encoded = [list(map(int, line.strip().split())) for line in f]

    assert encoded == [
        [1, 100, 1, 101, 2, 102, 3, 103],
        [1, 100, 2, 101, 3, 102, 4, 103],
    ]

    # clean up
    os.remove(input_file)
    os.remove(output_file)


def test_decode_from_file():
    input_file = "test_decode_from_file_input.txt"
    output_file = "test_decode_from_file_output.txt"

    with open(input_file, "w") as f:
        f.write("1 100 1 101 2 102 3 103\n1 100 2 101 3 102 4 103\n")

    tokenizer = RLETokenizer()
    decoded = tokenizer.decode_from_file(input_file, output_file)

    with open(output_file, "r") as f:
        decoded = [list(map(int, line.strip().split())) for line in f]

    assert decoded == [
        [0, 1, 2, 2, 3, 3, 3],
        [0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    ]

    # clean up
    os.remove(input_file)
    os.remove(output_file)
