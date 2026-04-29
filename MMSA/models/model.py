import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder


class MLPfusion(nn.Module):
    def __init__(
        self,
        orig_dim,
        output_dim=1,
        proj_dim=128,
        num_heads=5,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.15,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.2,
    ):
        super(MLPfusion, self).__init__()

        self.orig_dim = orig_dim
        self.out_dropout = out_dropout
        self.num_mod = len(orig_dim)
        total_dim = sum(orig_dim)

        self.input_norm = nn.LayerNorm(total_dim)
        self.fc1 = nn.Linear(total_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.fc3 = nn.Linear(proj_dim, proj_dim)
        self.hidden_norm = nn.LayerNorm(proj_dim)
        self.out = nn.Linear(proj_dim, output_dim)

    def _masked_mean_pool(self, x):
        # x: [B, T, D], where padded steps are zero vectors.
        mask = (x.abs().sum(dim=-1) > 0).float()  # [B, T]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def forward(self, x):
        pooled_features = [self._masked_mean_pool(modal) for modal in x]
        fused = torch.cat(pooled_features, dim=-1)
        fused = self.input_norm(fused)

        hidden = F.relu(self.fc1(fused))
        hidden = F.dropout(hidden, p=self.out_dropout, training=self.training)
        hidden_res = F.relu(self.fc2(hidden))
        hidden_res = F.dropout(hidden_res, p=self.out_dropout, training=self.training)
        hidden = self.hidden_norm(hidden + hidden_res)
        hidden = F.relu(self.fc3(hidden))
        hidden = F.dropout(hidden, p=self.out_dropout, training=self.training)

        # Bound prediction range to sentiment scale [-3, 3] for stable DG regression.
        output = 3.0 * torch.tanh(self.out(hidden))
        return output, hidden


class Earlyfusion(nn.Module):
    def __init__(
        self,
        orig_dim,
        output_dim=1,
        proj_dim=40,
        num_heads=5,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.15,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.2,
    ):
        super(Earlyfusion, self).__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout

        # Projection Layers
        self.proj = nn.ModuleList(
            [
                nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
                for i in range(self.num_mod)
            ]
        )

        # Fusion
        self.fusion = TransformerEncoder(
            embed_dim=proj_dim,
            num_heads=self.num_heads,
            layers=self.layers,
            attn_dropout=self.attn_dropout,
            res_dropout=self.res_dropout,
            relu_dropout=self.relu_dropout,
            embed_dropout=self.embed_dropout,
        )

        # Output layers
        self.out_layer_proj1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer_proj2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer = nn.Linear(self.proj_dim, output_dim)

    def forward(self, x):
        """
        dimension [batch_size, seq_len, n_features]
        """
        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)

        feature = torch.cat(x)
        last_hs = self.fusion(feature)[0]
        # A residual block
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                F.relu(self.out_layer_proj1(last_hs)),
                p=self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, last_hs


class Latefusion(nn.Module):
    def __init__(
        self,
        orig_dim,
        output_dim=1,
        proj_dim=40,
        num_heads=5,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.15,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.2,
    ):
        super(Latefusion, self).__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout

        # Projection Layers
        self.proj = nn.ModuleList(
            [
                nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
                for i in range(self.num_mod)
            ]
        )

        # Encoders
        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    embed_dim=proj_dim,
                    num_heads=self.num_heads,
                    layers=self.layers,
                    attn_dropout=self.attn_dropout,
                    res_dropout=self.res_dropout,
                    relu_dropout=self.relu_dropout,
                    embed_dropout=self.embed_dropout,
                )
                for _ in range(self.num_mod)
            ]
        )

        # Output layers
        self.out_layer_proj0 = nn.Linear(3 * self.proj_dim, self.proj_dim)
        self.out_layer_proj1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer_proj2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer = nn.Linear(self.proj_dim, output_dim)

    def forward(self, x):
        """
        dimension [batch_size, seq_len, n_features]
        """
        hs = list()

        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)
            h_tmp = self.encoders[i](x[i])
            hs.append(h_tmp[0])

        last_hs_out = torch.cat(hs, dim=-1)
        # A residual block
        last_hs = F.relu(self.out_layer_proj0(last_hs_out))
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                F.relu(self.out_layer_proj1(last_hs)),
                p=self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, last_hs_out
