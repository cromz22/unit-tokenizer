from bpe_tokenizer import BPETokenizer


def main():
    tokenizer = BPETokenizer()
    tokenizer.fit(train_data=[1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6], target_vocab_size=10)


if __name__ == "__main__":
    main()
