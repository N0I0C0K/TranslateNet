from typing import Iterable


def main():
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


main()
