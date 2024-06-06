import os

from tokenizers import PackBitsTokenizer


def test_encode():
    tokenizer = PackBitsTokenizer()
    encoded = tokenizer._encode([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6])
    assert encoded == [4, 100, 2, 101, 6, 102, 0, 4, 103, 104, 105, 106]


def test_decode():
    tokenizer = PackBitsTokenizer()
    decoded = tokenizer._decode([4, 100, 2, 101, 6, 102, 0, 4, 103, 104, 105, 106])
    assert decoded == [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6]


def test_batch_encode():
    tokenizer = PackBitsTokenizer()
    encoded = tokenizer.encode([[0, 0, 0, 0, 1, 1], [2, 2, 2, 2, 2, 2, 3, 4, 5, 6]])
    assert encoded == [[4, 100, 2, 101], [6, 102, 0, 4, 103, 104, 105, 106]]


def test_batch_decode():
    tokenizer = PackBitsTokenizer()
    decoded = tokenizer.decode([[4, 100, 2, 101], [6, 102, 0, 4, 103, 104, 105, 106]])
    assert decoded == [[0, 0, 0, 0, 1, 1], [2, 2, 2, 2, 2, 2, 3, 4, 5, 6]]


def test_encode_from_file():
    input_file = "test_encode_from_file_input.txt"
    output_file = "test_encode_from_file_output.txt"

    with open(input_file, "w") as f:
        f.write("0 0 0 0 1 1\n2 2 2 2 2 2 3 4 5 6\n")

    tokenizer = PackBitsTokenizer()
    tokenizer.encode_from_file(input_file, output_file)

    with open(output_file, "r") as f:
        encoded = [list(map(int, line.strip().split())) for line in f]

    assert encoded == [[4, 100, 2, 101], [6, 102, 0, 4, 103, 104, 105, 106]]

    # clean up
    os.remove(input_file)
    os.remove(output_file)


def test_decode_from_file():
    input_file = "test_decode_from_file_input.txt"
    output_file = "test_decode_from_file_output.txt"

    with open(input_file, "w") as f:
        f.write("4 100 2 101\n6 102 0 4 103 104 105 106\n")

    tokenizer = PackBitsTokenizer()
    tokenizer.decode_from_file(input_file, output_file)

    with open(output_file, "r") as f:
        decoded = [list(map(int, line.strip().split())) for line in f]

    assert decoded == [[0, 0, 0, 0, 1, 1], [2, 2, 2, 2, 2, 2, 3, 4, 5, 6]]

    # clean up
    os.remove(input_file)
    os.remove(output_file)
