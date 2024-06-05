from tokenizers import RLETokenizer


def test_encode():
    tokenizer = RLETokenizer()
    encoded = tokenizer.encode([0, 1, 2, 2, 3, 3, 3])
    assert encoded == [1, 100, 1, 101, 2, 102, 3, 103]
