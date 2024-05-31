from bpe_tokenizer import BPETokenizer


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
    import os

    os.remove("test_save_and_load.json")


def test_batch_encode_decode():
    tokenizer = BPETokenizer()
    tokenizer.fit(
        train_data=[[0, 1, 0, 1, 2, 0, 1, 2, 3], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]],
        target_vocab_size=10,
    )
    encoded = tokenizer.encode([[0, 1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2, 3]])
    assert encoded == [[6, 9, 5], [7, 8]]
    decoded = tokenizer.decode(encoded)
    assert decoded == [[0, 1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2, 3]]


def test_file_input():
    with open("test_file_input.txt", "w") as f:
        f.write("0 1 0 1 2 0 1 2 3\n0 1 2 3 4 0 1 2 3 4 5\n")

    tokenizer = BPETokenizer()
    tokenizer.fit_from_file("test_file_input.txt", target_vocab_size=10)

    assert tokenizer.merges == {(0, 1): 6, (6, 2): 7, (7, 3): 8, (8, 4): 9}

    # clean up
    import os

    os.remove("test_file_input.txt")


def main():
    # test_fit()
    # test_encode()
    # test_decode()
    # test_arbitrary_onset()
    # test_save_and_load()
    # test_batch_encode_decode()
    test_file_input()


if __name__ == "__main__":
    main()
