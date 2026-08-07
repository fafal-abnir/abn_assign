import torch
from tabm import TabM
from torch import nn

from deep_learning.config import DeepTrainingConfig


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        categorical_cardinalities: list[int],
        d_token: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.numeric_weight = nn.Parameter(torch.empty(n_numeric, d_token))
        self.numeric_bias = nn.Parameter(torch.empty(n_numeric, d_token))
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in categorical_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))
        nn.init.normal_(self.numeric_weight, std=0.02)
        nn.init.normal_(self.numeric_bias, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        numeric_tokens = (
            numeric.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
            + self.numeric_bias.unsqueeze(0)
        )
        categorical_tokens = torch.stack(
            [embedding(categorical[:, index]) for index, embedding in enumerate(self.category_embeddings)],
            dim=1,
        )
        cls = self.cls_token.expand(numeric.shape[0], -1, -1)
        tokens = torch.cat([cls, numeric_tokens, categorical_tokens], dim=1)
        return self.head(self.transformer(tokens)[:, 0]).squeeze(-1)


def create_deep_model(
    config: DeepTrainingConfig,
    n_numeric: int,
    categorical_cardinalities: list[int],
) -> nn.Module:
    if config.model_name == "ft_transformer":
        return FTTransformer(
            n_numeric,
            categorical_cardinalities,
            config.d_token,
            config.n_layers,
            config.n_heads,
            config.dropout,
        )
    if config.model_name == "tabm":
        return TabM.make(
            n_num_features=n_numeric,
            cat_cardinalities=categorical_cardinalities,
            d_out=1,
            d_block=config.d_token * 4,
            n_blocks=config.n_layers,
            dropout=config.dropout,
            k=config.tabm_k,
        )
    raise ValueError(f"Unknown model: {config.model_name}")
