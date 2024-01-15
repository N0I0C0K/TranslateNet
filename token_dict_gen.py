from collections import defaultdict
from typing import Iterable


def dict_gen():
    with (
        open("./data/news-commentary-v13.zh-en.en.split", "r") as en_file,
        open("./data/news-commentary-v13.zh-en.zh.split", "r") as cn_file,
        open("./data/token_dict.txt", "w") as token_file,
    ):
        word_dict: defaultdict[str, int] = defaultdict(int)
        max_len = 0
        for en_line in en_file:
            en_word = en_line.strip().split("/")
            for k in en_word:
                word_dict[k] += 1
            max_len = max(max_len, len(en_word))

        for cn_line in cn_file:
            cn_word = cn_line.strip().split("/")
            for k in cn_word:
                word_dict[k] += 1
            max_len = max(max_len, len(cn_word))

        a = 0
        for k in word_dict:
            if word_dict[k] >= 3:
                token_file.write(k + "\n")
                a += 1

        print(f"total token size:{a}\ntoken len:{max_len}")


def token_gen():
    with (
        open("./data/news-commentary-v13.zh-en.en.split", "r") as en_file,
        open("./data/news-commentary-v13.zh-en.zh.split", "r") as cn_file,
        open("data/token_dict.txt", "r") as word_dict_file,
        open("data/tokens.txt", "w") as token_file,
    ):
        idx2word = ["<bos>", "<eos>", "<pad>", "<unk>"]
        word2idx = {v: k for k, v in enumerate(idx2word)}
        i = len(idx2word)
        for line in word_dict_file:
            l = line.removesuffix("\n")
            idx2word.append(l)
            if l in word2idx:
                raise ValueError(l)
            word2idx[l] = i
            i += 1

        def words2token(words: list[str]) -> str:
            return ",".join(map(lambda x: str(word2idx.get(x, 3)), words))

        while True:
            en_line, cn_line = (
                en_file.readline().removesuffix("\n"),
                cn_file.readline().removesuffix("\n"),
            )
            if not en_line or not cn_line:
                break
            en_token = words2token(en_line.split("/"))
            cn_token = words2token(cn_line.split("/"))

            token_file.write(en_token + "\n")
            token_file.write(cn_token + "\n")

    def token2words(token: Iterable[int]) -> str:
        return " ".join(map(lambda x: idx2word[x], token))

    while (s := input(">")) != "exit":
        print(token2words(map(lambda x: int(x), s.split(","))))


dict_gen()
token_gen()
