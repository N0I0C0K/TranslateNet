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
        token_file: str = "./tokens.txt",
        dic_path: str = "./word_dic.txt",
        max_lines: int = -1,
        seq_len: int = 48
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
            self.load_tokens(token_file)
        else:
            raise ValueError

    def load_tokens(self, token_file: str):
        with open(token_file, "r", encoding="utf-8") as file:
            while True:
                en, cn = file.readline().strip(), file.readline().strip()
                if not en or not cn:
                    break
                self.data.append(
                    (list(map(int, en.split(","))), list(map(int, cn.split(","))))
                )
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

    def token2word(self, token: list[int]) -> str:
        return " ".join(self.idx2word[x] for x in token)

    def padding_token(self, token: list[int]) -> Tensor:
        res = [0]
        res.extend(token[:46])
        res.append(1)
        res.extend(2 for _ in range(self.seq_len - len(res)))
        return LongTensor(res).to(self.device)

    def load_dic(self, dic_path: str):
        with open(dic_path, "r", encoding="utf-8") as file:
            self.idx2word.clear()
            for i, s in enumerate(file):
                s = s.strip()
                self.idx2word.append(s)
                self.word2idx[s] = i

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        en, cn = self.data[index]
        return (self.padding_token(cn), self.padding_token(en))
        # if index % 2 == 0:
        #     return (self.padding_token(en), self.padding_token(cn))
        # else:
