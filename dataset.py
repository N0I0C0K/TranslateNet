import os
from torch import Tensor, LongTensor, device
from torch.utils.data import Dataset

from typing import Iterable, Literal

import spacy
from spacy.language import Language


def load_dict(
    en_dict_path: str, cn_dict_path: str
) -> tuple[list[str], dict[str, int], list[str], dict[str, int]]:
    with (
        open(en_dict_path, "r", encoding="utf-8") as en_dict_file,
        open(cn_dict_path, "r", encoding="utf-8") as cn_dict_file,
    ):
        en_idx2word = ["<bos>", "<eos>", "<pad>", "<unk>"]
        en_word2idx = {v: k for k, v in enumerate(en_idx2word)}
        i = len(en_idx2word)
        for line in en_dict_file:
            l = line.removesuffix("\n")
            if not l:
                continue
            en_idx2word.append(l)
            en_word2idx[l] = i
            i += 1

        cn_idx2word = ["<bos>", "<eos>", "<pad>", "<unk>"]
        cn_word2idx = {v: k for k, v in enumerate(cn_idx2word)}
        i = len(cn_idx2word)
        for line in cn_dict_file:
            l = line.removesuffix("\n")
            if not l:
                continue
            cn_idx2word.append(l)
            cn_word2idx[l] = i
            i += 1

        return en_idx2word, en_word2idx, cn_idx2word, cn_word2idx


class TranslationData(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        device: device,
        *,
        token_file: str = "./data/tokens.txt",
        en_dic_path: str = "./data/en_token_dict.txt",
        cn_dic_path: str = "./data/cn_token_dict.txt",
        max_lines: int = -1,
        seq_len: int = 128,
    ) -> None:
        super().__init__()

        self.device = device
        self.data: list[tuple[list[int], list[int]]] = []
        self.seq_len = seq_len

        self._cn_tokenizer: Language | None = None

        (
            self.en_idx2word,
            self.en_word2idx,
            self.cn_idx2word,
            self.cn_word2idx,
        ) = load_dict(en_dic_path, cn_dic_path)

        self.load_tokens(token_file, max_lines)

        print(
            self.token2word(self.data[-1][0], "en"),
            self.token2word(self.data[-1][1], "zh"),
        )

    def load_tokens(self, token_file: str, max_lines: int = -1):
        with open(token_file, "r", encoding="utf-8") as file:
            while max_lines <= -1 or max_lines != 0:
                en, cn = file.readline().strip(), file.readline().strip()
                if not en or not cn:
                    break
                self.data.append(
                    (list(map(int, en.split(","))), list(map(int, cn.split(","))))
                )
                max_lines -= 1

    @property
    def cn_tokenizer(self) -> Language:
        if self._cn_tokenizer is None:
            spacy.prefer_gpu()  # type: ignore
            self._cn_tokenizer = spacy.load("zh_core_web_md")
        return self._cn_tokenizer

    def word2token(self, words: str) -> Tensor:
        return self.padding_token(self.word2rawtoken(words))

    def word2rawtoken(self, words: str) -> list[int]:
        return list(self.cn_word2idx.get(x.text, 3) for x in self.cn_tokenizer(words))

    def token2word(self, token: Iterable[int], lan: Literal["zh", "en"]) -> str:
        return (
            " ".join(self.cn_idx2word[x] for x in token)
            if lan == "zh"
            else " ".join(self.en_idx2word[x] for x in token)
        )

    def padding_token(self, token: list[int]) -> Tensor:
        res = [0]
        res.extend(token[: self.seq_len - 2])
        res.append(1)
        res.extend(2 for _ in range(self.seq_len - len(res)))
        return LongTensor(res).to(self.device)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        en, cn = self.data[index]
        if len(en) > self.seq_len - 2 or len(cn) > self.seq_len - 2:
            en, cn = [], []
        return (self.padding_token(cn), self.padding_token(en))
        # if index % 2 == 0:
        #     return (self.padding_token(en), self.padding_token(cn))
        # else:


test_words = [
    [
        89,
        21,
        94,
        95,
        15,
        96,
        97,
        25,
        98,
        3,
        99,
        100,
        17,
        82,
        93,
        101,
        21,
        68,
        102,
        103,
        104,
        17,
        105,
        106,
        53,
        107,
        108,
        25,
        109,
        110,
        111,
        112,
        25,
        113,
        114,
        36,
    ],  # 欧洲 在 避免 债务 和 捍卫 欧元 的 名义 <unk> 变得 谨慎 ， 而 美国 已经 在 许多 方面 行动 起来 ， 以 利用 这 一 理想 的 时机 来 实行 急需 的 结构性 改革 。
    [123, 21, 2036, 491, 413],  # 我在这里等你
]
