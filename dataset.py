import os
from torch import Tensor, LongTensor, device
from torch.utils.data import Dataset


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

        if os.path.exists(dic_path) and os.path.exists(token_file):
            self.load_dic(dic_path)
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
        return len(self.data) * 2

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        en, cn = self.data[index // 2]
        if index % 2 == 0:
            return (self.padding_token(en), self.padding_token(cn))
        else:
            return (self.padding_token(cn), self.padding_token(en))
