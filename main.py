import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader

import time
import os

from rich.progress import Progress

from model import TranslationNet
from dataset import TranslationData, test_words

batch_size = 12
lr = 0.00011
seed = torch.manual_seed(417)


class Translator:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0")

        self.dataset = TranslationData(
            self.device,
            seq_len=64,
            max_lines=-1,
            token_file=f"./mid_data/tokens.txt",
            cn_dic_path="./mid_data/cn_token_dict.txt",
            en_dic_path="./mid_data/en_token_dict.txt",
        )

        train_data, vaild_data = random_split(
            self.dataset,
            [len(self.dataset) - 2000, 2000],
            generator=seed,
        )
        self.data_split_len = 5
        self.train_data_list = random_split(
            train_data,
            [1 / self.data_split_len for _ in range(self.data_split_len)],
            generator=seed,
        )
        # self.train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        self.vail_loader = DataLoader(vaild_data, shuffle=True, batch_size=batch_size)

        # t_data = next(iter(self.vail_loader))
        # print(self.dataset.token2word(t_data[0][0], "zh"))

        self.net = TranslationNet(
            len(self.dataset.cn_word2idx),
            len(self.dataset.en_word2idx),
            self.device,
            hidden_size=1024,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 128
        )

        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=2)
        self.progress = Progress()

        self.last_loss = 0
        self.epoch = 0

        self.loaded_checkpoint_file = ""
        import glob

        files = glob.glob("checkpoint-*.pth")
        for i, file in enumerate(files):
            print(f"{i}> {file}")
        if files:
            t = input(
                "choose check point to load, default is the last one, n to unload>"
            )
            if t == "":
                t = -1
            if t != "n":
                self.load_checkpoint(files[int(t)])

    def gen_checkpoint_name(self) -> str:
        return f"checkpoint-mid-{time.strftime('%m-%d-%H%M')}-{self.epoch}-{self.last_loss:.3f}.pth"

    def save_checkpoint(self):
        file_name = self.loaded_checkpoint_file or self.gen_checkpoint_name()
        with open(file_name, "wb") as file:
            torch.save(
                {
                    "net_state": self.net.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                },
                file,
            )
        tgt_file = self.gen_checkpoint_name()
        os.rename(file_name, tgt_file)
        print(f"save check point to {tgt_file}")
        self.loaded_checkpoint_file = tgt_file

    def load_checkpoint(self, file: str):
        ckpt = torch.load(file)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt["epoch"]

        self.loaded_checkpoint_file = file
        self.optimizer_scheduler.last_epoch = self.epoch
        print(f"loaded check point: {file}, epoch: {self.epoch}")

    def forward_net(self, src: Tensor, tgt: Tensor):
        src, tgt = src.to(self.device), tgt.to(self.device)
        src_mask = torch.ones_like(src).float().masked_fill(src == 2, float("-inf"))

        dec_tgt = tgt[:, :-1]
        dec_tgt_mask = (
            torch.ones_like(dec_tgt).float().masked_fill(dec_tgt == 2, float("-inf"))
        )
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            dec_tgt.size(1), self.device
        )

        out = self.net.forward(src, dec_tgt, tgt_mask, src_mask, dec_tgt_mask)
        return out

    def train_epoch(self):
        self.net.train()
        train_loader = DataLoader(
            self.train_data_list[self.epoch % self.data_split_len],
            shuffle=True,
            batch_size=batch_size,
        )
        train_progress = self.progress.add_task(
            description="Train Epoch", total=len(train_loader)
        )
        # ignore <pad> which index is 2
        loss_f = self.loss_f  # torch.nn.CrossEntropyLoss(ignore_index=2)

        voacb_size = len(self.dataset.en_word2idx)
        len_data = len(train_loader)
        loss_all = 0
        for i, (src, tgt) in enumerate(train_loader):
            out = self.forward_net(src, tgt)
            loss = loss_f.forward(out.reshape(-1, voacb_size), tgt[:, 1:].flatten())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.progress.update(
                train_progress,
                advance=1,
                description=f"{i}/{len_data} loss={loss.item():.4f}",
            )
            loss_all += loss.item()
        self.optimizer_scheduler.step()
        self.progress.remove_task(train_progress)
        self.progress.print(
            f"train epoch={self.epoch} average loss={loss_all/len_data:.4f} lr={self.optimizer_scheduler.get_lr()}"
        )
        self.last_loss = loss_all / len_data

    def evaluation(self):
        self.net.eval()

        loss_f = self.loss_f
        voacb_size = len(self.dataset.en_word2idx)

        loss_a = 0
        with torch.no_grad():
            for i, (src, tgt) in enumerate(self.vail_loader):
                out = self.forward_net(src, tgt)
                loss = loss_f.forward(out.reshape(-1, voacb_size), tgt[:, 1:].flatten())
                loss_a += loss.item()

        self.progress.print(
            f"Validation: epoch={self.epoch} avg loss={loss_a/len(self.vail_loader):.4f}"
        )

    def training(self, train_epoch_nums: int = 36):
        self.progress.start()
        training_all = self.progress.add_task(
            description=f"epoch={self.epoch} lr={self.optimizer_scheduler.get_lr()}",
            total=train_epoch_nums,
        )

        for i in range(train_epoch_nums):
            self.progress.update(
                training_all,
                advance=1,
                description=f"epoch={self.epoch} lr={self.optimizer_scheduler.get_lr()}",
            )
            self.train_epoch()
            self.evaluation()
            self.epoch += 1
            self.save_checkpoint()
            for words in test_words:
                print(" ")
                print(f"src: {self.dataset.token2word(words, 'zh')}")
                print(f"tgt: {self.translate_token(words)}")

    def translate_token(self, src_token: list[int]) -> str:
        self.net.eval()
        start_words_token = [0]
        src = self.dataset.padding_token(src_token).unsqueeze(0)
        tgt = torch.LongTensor([start_words_token]).to(self.device)
        memo = self.net.encode(src)
        res = []
        for i in range(48):
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), self.device
            )
            out = self.net.decode(tgt, memo, tgt_mask)
            next_word = out.argmax(2)
            if next_word[0][-1] == 1:
                break
            # tgt = torch.cat((tgt, ))
            res.append(next_word[0][-1].item())
            tgt = torch.cat((tgt, next_word[:, -1:]), 1)

        return self.dataset.token2word(res, "en")

    def translate(self, src_word: str) -> str:
        return self.translate_token(self.dataset.word2rawtoken(src_word))


def main():
    trainer = Translator()
    trainer.training(128)
    # while (s := input(">")) != "exit":
    #     print(trainer.translate(s))


if __name__ == "__main__":
    main()
