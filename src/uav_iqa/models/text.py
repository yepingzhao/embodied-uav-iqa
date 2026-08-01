from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionTextEncoder(nn.Module):
    """Character-level CNN text encoder for short VQA question text.

    Uses multiple 1D convolution kernel sizes (3, 4, 5) over character
    embeddings followed by max-over-time pooling — a classic Char-CNN
    architecture (Zhang et al., 2015) adapted for lightweight deployment.

    Vocab: 95 printable ASCII characters + PAD(0) + UNK(1) = 97 entries.
    Max sequence length is truncated to ``max_len``.
    """

    _PRINTABLE = (
        " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789.,;:!?\"'()-[]{}@#$%^&*+=/\\<>_`~|"
    )
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(
        self,
        text_dim: int = 128,
        char_embed_dim: int = 64,
        num_filters: int = 64,
        max_len: int = 256,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.max_len = max_len

        self.vocab_size = len(self._PRINTABLE) + 2  # + PAD + UNK
        self.char_to_idx = {ch: i + 2 for i, ch in enumerate(self._PRINTABLE)}

        self.char_embed = nn.Embedding(self.vocab_size, char_embed_dim, padding_idx=self.PAD_IDX)

        self.conv3 = nn.Conv1d(char_embed_dim, num_filters, 3, padding=1)
        self.conv4 = nn.Conv1d(char_embed_dim, num_filters, 4, padding=2)
        self.conv5 = nn.Conv1d(char_embed_dim, num_filters, 5, padding=2)

        self.proj = nn.Linear(num_filters * 3, text_dim)

    def _tokenize(self, texts: List[str]) -> torch.Tensor:
        """Convert a list of strings to a padded (B, max_len) tensor of char indices."""
        batch_rows = []
        for text in texts:
            row = [self.char_to_idx.get(ch, self.UNK_IDX) for ch in text[: self.max_len]]
            batch_rows.append(row)
        max_actual = max((max(len(r) for r in batch_rows), 1)) if batch_rows else 1
        padded = [r + [self.PAD_IDX] * (max_actual - len(r)) for r in batch_rows]
        return torch.tensor(padded, dtype=torch.long)

    def forward(self, texts: List[str]) -> torch.Tensor:
        token_ids = self._tokenize(texts).to(self.char_embed.weight.device)
        x = self.char_embed(token_ids)  # (B, max_len, char_embed_dim)
        x = x.transpose(1, 2)  # (B, char_embed_dim, max_len)

        c3 = F.relu(self.conv3(x)).max(dim=2).values  # (B, num_filters)
        c4 = F.relu(self.conv4(x)).max(dim=2).values
        c5 = F.relu(self.conv5(x)).max(dim=2).values

        combined = torch.cat([c3, c4, c5], dim=1)  # (B, num_filters*3)
        return self.proj(combined)  # (B, text_dim)
