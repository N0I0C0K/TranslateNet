from collections import defaultdict
from typing import Iterable

target_path = "mid_data"


def dict_gen():
    with (
        open("./data/news-commentary-v13.zh-en.en.split", "r") as en_file,
        open("./data/news-commentary-v13.zh-en.zh.split", "r") as cn_file,
        open(f"./{target_path}/en_token_dict.txt", "w") as en_token_file,
        open(f"./{target_path}/cn_token_dict.txt", "w") as cn_token_file,
    ):
        en_dict: defaultdict[str, int] = defaultdict(int)
        cn_dict: defaultdict[str, int] = defaultdict(int)
        len_dict: defaultdict[int, int] = defaultdict(int)
        for en_line in en_file:
            en_word = en_line.strip().split("/")
            for k in en_word:
                en_dict[k] += 1
            len_dict[len(en_word)] += 1

        for cn_line in cn_file:
            cn_word = cn_line.strip().split("/")
            for k in cn_word:
                cn_dict[k] += 1
            len_dict[len(cn_word)] += 1

        en_total, en_miss = 0, 0
        cn_total, cn_miss = 0, 0
        for k in en_dict:
            if len(k) > 0 and en_dict[k] >= 2:
                en_token_file.write(k + "\n")
                en_total += 1
            else:
                en_miss += en_dict[k]

        for k in cn_dict:
            if len(k) > 0 and cn_dict[k] >= 2:
                cn_token_file.write(k + "\n")
                cn_total += 1
            else:
                cn_miss += cn_dict[k]

        print(
            f"en token size:{(en_total, en_miss)}\ncn token size:{(cn_total, cn_miss)}"
        )
        # lenl = [0 for _ in range(200)]
        # for k in len_dict:
        #     lenl[k] = len_dict[k]

        # all_token = sum(lenl)
        # for i in range(1, 200):
        #     lenl[i] += lenl[i - 1]

        # for i in range(200):
        #     print(f"<={i} num:{lenl[i]} {lenl[i]/all_token:.4f}")


def load_dict() -> tuple[list[str], dict[str, int], list[str], dict[str, int]]:
    with (
        open(f"./{target_path}/en_token_dict.txt", "r") as en_dict_file,
        open(f"./{target_path}/cn_token_dict.txt", "r") as cn_dict_file,
    ):
        en_idx2word = ["<bos>", "<eos>", "<pad>", "<unk>"]
        en_word2idx = {v: k for k, v in enumerate(en_idx2word)}
        i = len(en_idx2word)
        for line in en_dict_file:
            l = line.removesuffix("\n")
            if not l:
                raise
            en_idx2word.append(l)
            en_word2idx[l] = i
            i += 1

        cn_idx2word = ["<bos>", "<eos>", "<pad>", "<unk>"]
        cn_word2idx = {v: k for k, v in enumerate(cn_idx2word)}
        i = len(cn_idx2word)
        for line in cn_dict_file:
            l = line.removesuffix("\n")
            if not l:
                raise
            cn_idx2word.append(l)
            cn_word2idx[l] = i
            i += 1

        return en_idx2word, en_word2idx, cn_idx2word, cn_word2idx


def token_gen():
    with (
        open("./data/news-commentary-v13.zh-en.en.split", "r") as en_file,
        open("./data/news-commentary-v13.zh-en.zh.split", "r") as cn_file,
        open(f"./{target_path}/tokens.txt", "w") as token_file,
    ):
        en_idx2word, en_word2idx, cn_idx2word, cn_word2idx = load_dict()

        def en_words2token(words: list[str]) -> str:
            return ",".join(map(lambda x: str(en_word2idx.get(x, 3)), words))

        def cn_words2token(words: list[str]) -> str:
            return ",".join(map(lambda x: str(cn_word2idx.get(x, 3)), words))

        while True:
            en_line, cn_line = (
                en_file.readline().removesuffix("\n"),
                cn_file.readline().removesuffix("\n"),
            )
            if not en_line or not cn_line:
                break
            en_token = en_words2token(en_line.split("/"))
            cn_token = cn_words2token(cn_line.split("/"))

            token_file.write(en_token + "\n")
            token_file.write(cn_token + "\n")

    def token2words(token: Iterable[int]) -> str:
        return " ".join(map(lambda x: cn_idx2word[x], token))

    while (s := input(">")) != "exit":
        print(token2words(map(lambda x: int(x), s.split(","))))


dict_gen()
token_gen()
