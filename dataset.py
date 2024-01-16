import os
from torch import Tensor, LongTensor, device
from torch.utils.data import Dataset

from typing import Iterable

import spacy
from spacy.language import Language


class TranslationData(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        device: device,
        *,
        token_file: str = "./data/tokens.txt",
        dic_path: str = "./data/token_dict.txt",
        max_lines: int = -1,
        seq_len: int = 128
    ) -> None:
        super().__init__()

        self.device = device
        self.idx2word: list[str] = ["<bos>", "<eos>", "<pad>", "<unk>"]
        self.word2idx: dict[str, int] = {}
        self.data: list[tuple[list[int], list[int]]] = []
        self.seq_len = seq_len
        self.vocab_size = 0
        self._cn_tokenizer: Language | None = None

        if os.path.exists(dic_path) and os.path.exists(token_file):
            self.load_dic(dic_path)
            self.load_tokens(token_file, max_lines)
        else:
            raise ValueError
        print(self.token2word(self.data[-1][0]), self.token2word(self.data[-1][1]))

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
        self.vocab_size = len(self.word2idx)

    @property
    def cn_tokenizer(self) -> Language:
        if self._cn_tokenizer is None:
            spacy.prefer_gpu()  # type: ignore
            self._cn_tokenizer = spacy.load("zh_core_web_md")
        return self._cn_tokenizer

    def word2token(self, words: str) -> Tensor:
        return self.padding_token(self.word2rawtoken(words))

    def word2rawtoken(self, words: str) -> list[int]:
        return list(self.word2idx[x.text] for x in self.cn_tokenizer(words))

    def token2word(self, token: Iterable[int]) -> str:
        return " ".join(self.idx2word[x] for x in token)

    def padding_token(self, token: list[int]) -> Tensor:
        res = [0]
        res.extend(token[: self.seq_len - 2])
        res.append(1)
        res.extend(2 for _ in range(self.seq_len - len(res)))
        return LongTensor(res).to(self.device)

    def load_dic(self, dic_path: str):
        with open(dic_path, "r", encoding="utf-8") as file:
            self.idx2word.clear()
            self.idx2word.extend(["<bos>", "<eos>", "<pad>", "<unk>"])
            i = 4
            for s in file:
                s = s.removesuffix("\n")
                self.idx2word.append(s)
                self.word2idx[s] = i
                i += 1

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
        37119,
        110,
        37120,
        37121,
        36748,
        37122,
        37123,
        37124,
        37125,
        37126,
        37127,
        37128,
        37129,
        37130,
        37131,
        37132,
        37133,
        37134,
        37135,
        37136,
        37137,
        37138,
        37139,
        37140,
        37141,
        37142,
        37143,
        37134,
        37144,
        37145,
    ],  # 巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。
    [37232, 37130, 39069, 37586, 37516],  # 我在这里等你
]
