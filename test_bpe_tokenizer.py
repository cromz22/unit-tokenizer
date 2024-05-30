from bpe_tokenizer import BPETokenizer


def test_fit():
    tokenizer = BPETokenizer()
    tokenizer.fit(train_data=[1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6], target_vocab_size=10)


def test_encode():
    tokenizer = BPETokenizer()
    tokenizer.fit(train_data=[1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6], target_vocab_size=10)
    encoded = tokenizer.encode([1, 2, 3, 4, 5, 6])
    assert encoded == [9, 6]


def main():
    # test_fit()
    test_encode()


if __name__ == "__main__":
    main()
