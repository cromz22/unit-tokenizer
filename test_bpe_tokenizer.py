from bpe_tokenizer import BPETokenizer


def test_fit():
    tokenizer = BPETokenizer()
    tokenizer.fit(train_data=[0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5], target_vocab_size=10)


def test_encode():
    tokenizer = BPETokenizer()
    tokenizer.fit(train_data=[0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5], target_vocab_size=10)
    encoded = tokenizer.encode([0, 1, 0, 1, 2, 3, 4, 5])
    assert encoded == [6, 9, 5]

def test_decode():
    tokenizer = BPETokenizer()
    tokenizer.fit(train_data=[0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5], target_vocab_size=10)
    encoded = tokenizer.encode([0, 1, 0, 1, 2, 3, 4, 5])
    decoded = tokenizer.decode(encoded)
    assert decoded == [0, 1, 0, 1, 2, 3, 4, 5]

def main():
    # test_fit()
    # test_encode()
    test_decode()


if __name__ == "__main__":
    main()
