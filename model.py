import math
import torch
from torch import nn as nn
from torch import Tensor, device


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, *, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        self.eq = math.sqrt(self.emb_size)

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * self.eq


class TranslationNet(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        device: device,
        *,
        embed_size: int = 512,
        n_head=8,
        n_layer=4,
        hidden_size=512
    ) -> None:
        super().__init__()
        self.src_token_embed = TokenEmbedding(src_vocab_size, embed_size).to(device)
        self.tgt_token_embed = TokenEmbedding(tgt_vocab_size, embed_size).to(device)

        self.transformer = nn.Transformer(
            embed_size,
            nhead=n_head,
            num_decoder_layers=n_layer,
            num_encoder_layers=n_layer,
            batch_first=True,
            dim_feedforward=hidden_size,
            device=device,
        )
        self.liner = nn.Linear(embed_size, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(embed_size).to(device)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ):
        src = self.positional_encoding(self.src_token_embed(src))

        tgt = self.positional_encoding(self.tgt_token_embed(tgt))

        out = self.transformer.forward(
            src,
            tgt,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        out = self.liner.forward(out)
        return out

    def encode(
        self, src: Tensor, *, noise: bool = True, noise_intensity: float = 0.3
    ) -> Tensor:
        embeded = self.src_token_embed.forward(src)
        if noise:
            embeded += torch.rand_like(embeded) * noise_intensity
        return self.transformer.encoder.forward(
            self.positional_encoding.forward(embeded),
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        return self.liner.forward(
            self.transformer.decoder.forward(
                self.positional_encoding.forward(self.tgt_token_embed(tgt)),
                memory,
                tgt_mask,
            )
        )
