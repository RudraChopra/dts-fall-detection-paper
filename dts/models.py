"""
Model definitions
=================
All PyTorch models used in the paper:
  - FallLSTM         2-layer LSTM on raw (T,34) keypoint sequences
  - FallGRU          2-layer GRU  on raw (T,34) keypoint sequences
  - FallTransformer  Lightweight Transformer encoder on raw sequences
  - DTSNet           Asymmetry-guided family attention on DTS-128 features

Sequence models expect (padded_seq, lengths) input.
DTSNet expects a (B, 128) DTS feature vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


# ──────────────────────────────────────────────────────────────────────────────
#  Sequence baselines
# ──────────────────────────────────────────────────────────────────────────────

class FallLSTM(nn.Module):
    """2-layer LSTM → final hidden state → binary logit."""

    def __init__(self, input_dim: int = 34, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.fc  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                     enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        return self.fc(h[-1]).squeeze(-1)          # (B,)


class FallGRU(nn.Module):
    """2-layer GRU → final hidden state → binary logit."""

    def __init__(self, input_dim: int = 34, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.fc  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                     enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.fc(h[-1]).squeeze(-1)


class FallTransformer(nn.Module):
    """
    Lightweight Transformer encoder.
    Mean-pools over valid frames → binary logit.
    """

    def __init__(self, input_dim: int = 34, d_model: int = 64,
                 nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, nlayers)
        self.fc   = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.d_model = d_model

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = (torch.arange(T, device=x.device).float()
               .unsqueeze(0).unsqueeze(-1) / 100.0)
        h   = self.proj(x) + pos.expand(B, T, self.d_model) * 0.01
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for i, l in enumerate(lengths):
            if l < T:
                mask[i, l:] = True
        out   = self.enc(h, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pool  = (out * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.fc(pool).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
#  DTS-Net
# ──────────────────────────────────────────────────────────────────────────────

class DTSNet(nn.Module):
    """
    Asymmetry-Guided Family Attention Network.

    Input : (B, 128) DTS feature vector split into 8 families of 16 elements.
    The 16th element of each family block (index 15, 31, …, 127) is the
    temporal asymmetry α for that primitive.

    Architecture
    ────────────
    1. Extract 8-dim α-vector from element 15 of each 16-dim block.
    2. Learn family attention: a = softmax(W_a · α + b_a)  ∈ ℝ⁸
    3. Re-weight: φ_attn = concat(a_i · φ^(i))  ∈ ℝ¹²⁸
    4. MLP: 128 → 64 → 32 → 1  (dropout 0.15 after first layer)
    5. Auxiliary head: predict mean(α) from hidden layer (MSE loss)

    Training loss: BCE(primary) + λ * MSE(auxiliary α prediction)
    """

    def __init__(self, n_fam: int = 8, fam_dim: int = 16,
                 h1: int = 64, h2: int = 32,
                 dropout: float = 0.15, lam_aux: float = 0.15):
        super().__init__()
        self.n_fam   = n_fam
        self.fam_dim = fam_dim
        self.lam_aux = lam_aux

        # α-vector index positions
        self.register_buffer(
            'alpha_cols',
            torch.tensor([i * fam_dim + (fam_dim - 1) for i in range(n_fam)])
        )

        # Attention gate
        self.attn = nn.Linear(n_fam, n_fam)

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_fam * fam_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),             nn.ReLU(),
            nn.Linear(h2, 1)
        )

        # Auxiliary α predictor
        self.aux = nn.Linear(h1, 1)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 128) DTS feature tensor

        Returns
        -------
        logit      : (B,)  classification logit
        attn       : (B,8) family attention weights
        alpha_pred : (B,)  auxiliary α prediction
        alpha_vec  : (B,8) input α vector
        """
        B = x.shape[0]
        av  = x[:, self.alpha_cols]                              # (B, 8)
        att = torch.softmax(self.attn(av), dim=-1)               # (B, 8)
        xw  = (x.view(B, self.n_fam, self.fam_dim)
               * att.unsqueeze(-1)).view(B, -1)                  # (B, 128)

        h1     = self.mlp[2](self.mlp[1](self.mlp[0](xw)))      # (B, h1)
        logit  = self.mlp[5](self.mlp[4](self.mlp[3](h1))).squeeze(-1)
        ap     = self.aux(h1).squeeze(-1)                        # (B,)

        return logit, att, ap, av

    def training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Combined BCE + auxiliary MSE loss."""
        logit, _, ap, av = self.forward(x)
        bce = F.binary_cross_entropy_with_logits(logit, y)
        mse = F.mse_loss(ap, av.mean(dim=-1))
        return bce + self.lam_aux * mse

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 2) probability tensor [p_nfall, p_fall]."""
        logit, _, _, _ = self.forward(x)
        p = torch.sigmoid(logit)
        return torch.stack([1 - p, p], dim=-1)

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Returns mean attention weight per family over the batch."""
        _, att, _, _ = self.forward(x)
        return att.mean(dim=0)                                   # (8,)
