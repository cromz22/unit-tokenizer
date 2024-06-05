from tokenizers import BPETokenizer
import os


def test_fit():
    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]],
        target_vocab_size=10,
    )
    assert tokenizer.merges == {(0, 1): 6, (6, 2): 7, (7, 3): 8, (8, 4): 9}


def test_encode():
    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]],
        target_vocab_size=10,
    )
    encoded = tokenizer.encode([[0, 1, 0, 1, 2, 3, 4, 5]])
    assert encoded[0] == [6, 9, 5]


def test_decode():
    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]],
        target_vocab_size=10,
    )
    encoded = tokenizer.encode([[0, 1, 0, 1, 2, 3, 4, 5]])
    decoded = tokenizer.decode(encoded)
    assert decoded[0] == [0, 1, 0, 1, 2, 3, 4, 5]


def test_arbitrary_onset():
    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6]],
        target_vocab_size=10,
    )
    encoded = tokenizer.encode([[1, 2, 1, 2, 3, 4, 5, 6]])
    decoded = tokenizer.decode(encoded)
    assert decoded[0] == [1, 2, 1, 2, 3, 4, 5, 6]


def test_save_and_load():
    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]],
        target_vocab_size=10,
    )
    tokenizer.save("test_save_and_load.json")
    tokenizer = None

    tokenizer = BPETokenizer()
    tokenizer.load("test_save_and_load.json")
    encoded = tokenizer.encode([[0, 1, 0, 1, 2, 3, 4, 5]])
    assert encoded[0] == [6, 9, 5]
    decoded = tokenizer.decode(encoded)
    assert decoded[0] == [0, 1, 0, 1, 2, 3, 4, 5]

    # clean up
    os.remove("test_save_and_load.json")


def test_batch_encode():
    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[0, 1, 0, 1, 2, 0, 1, 2, 3], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]],
        target_vocab_size=10,
    )
    encoded = tokenizer.encode([[0, 1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2, 3]])
    assert encoded == [[6, 9, 5], [7, 8]]


def test_batch_decode():
    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[0, 1, 0, 1, 2, 0, 1, 2, 3], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]],
        target_vocab_size=10,
    )
    encoded = [[6, 9, 5], [7, 8]]
    decoded = tokenizer.decode(encoded)
    assert decoded == [[0, 1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2, 3]]


def test_fit_from_file():
    train_data_file = "test_fit_from_file.txt"

    with open(train_data_file, "w") as f:
        f.write("0 1 0 1 2 0 1 2 3\n0 1 2 3 4 0 1 2 3 4 5\n")

    tokenizer = BPETokenizer()
    tokenizer.fit_from_file(train_data_file, target_vocab_size=10)

    assert tokenizer.merges == {(0, 1): 6, (6, 2): 7, (7, 3): 8, (8, 4): 9}

    # clean up
    os.remove(train_data_file)


def test_encode_from_file():
    input_file = "test_encode_from_file_input.txt"
    output_file = "test_encode_from_file_output.txt"

    with open(input_file, "w") as f:
        f.write("0 1 0 1 2 3 4 5\n0 1 2 0 1 2 3\n")

    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[0, 1, 0, 1, 2, 0, 1, 2, 3], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]],
        target_vocab_size=10,
    )
    tokenizer.encode_from_file(input_file, output_file)

    with open(output_file, "r") as f:
        encoded = [list(map(int, line.strip().split())) for line in f]

    assert encoded == [[6, 9, 5], [7, 8]]

    # clean up
    os.remove(input_file)
    os.remove(output_file)


def test_decode_from_file():
    input_file = "test_decode_from_file_input.txt"
    output_file = "test_decode_from_file_output.txt"

    with open(input_file, "w") as f:
        f.write("6 9 5\n7 8\n")

    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[0, 1, 0, 1, 2, 0, 1, 2, 3], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]],
        target_vocab_size=10,
    )
    tokenizer.decode_from_file(input_file, output_file)

    with open(output_file, "r") as f:
        decoded = [list(map(int, line.strip().split())) for line in f]

    assert decoded == [[0, 1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2, 3]]

    # clean up
    os.remove(input_file)
    os.remove(output_file)
