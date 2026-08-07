"""PyTorch Lightning classification and imbalance-aware losses."""

import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from deep_learning.config import DeepTrainingConfig


class FraudLightningModule(LightningModule):
    def __init__(
        self, model: nn.Module, config: DeepTrainingConfig, positive_weight: float
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.positive_weight = positive_weight
        self.validation_ap = BinaryAveragePrecision()
        self.validation_auc = BinaryAUROC()
        self.test_ap = BinaryAveragePrecision()
        self.test_auc = BinaryAUROC()

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        logits = self.model(numeric, categorical)
        return logits.squeeze(-1)

    def _member_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits if logits.ndim == 2 else logits.unsqueeze(1)

    def _probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self._member_logits(logits)).mean(dim=1)

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        member_logits = self._member_logits(logits)
        expanded_target = target.unsqueeze(1).expand_as(member_logits)
        if self.config.loss_name in ("weighted_bce", "weighted_bce+deviation"):
            weighted_bce = F.binary_cross_entropy_with_logits(
                member_logits,
                expanded_target,
                pos_weight=torch.tensor(self.positive_weight, device=self.device),
            )
            if self.config.loss_name == "weighted_bce":
                return weighted_bce
            deviation = self._deviation_loss(member_logits, expanded_target)
            return weighted_bce + self.config.deviation_weight * deviation
        element_loss = F.binary_cross_entropy_with_logits(
            member_logits, expanded_target, reduction="none"
        )
        probability = torch.sigmoid(member_logits)
        probability_true = torch.where(expanded_target.bool(), probability, 1 - probability)
        alpha = torch.where(
            expanded_target.bool(),
            self.config.focal_alpha,
            1 - self.config.focal_alpha,
        )
        return (alpha * (1 - probability_true).pow(self.config.focal_gamma) * element_loss).mean()

    def _deviation_loss(
        self, anomaly_scores: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Z-score deviation loss from Pang et al., KDD 2019, Equation 7."""
        reference = torch.randn(
            self.config.deviation_reference_samples,
            device=anomaly_scores.device,
            dtype=anomaly_scores.dtype,
        )
        reference_mean = reference.mean()
        reference_std = reference.std(unbiased=False).clamp_min(1e-6)
        deviation = (anomaly_scores - reference_mean) / reference_std
        normal_loss = deviation.abs()
        fraud_loss = F.relu(self.config.deviation_margin - deviation)
        return torch.where(target.bool(), fraud_loss, normal_loss).mean()

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_index: int) -> torch.Tensor:
        numeric, categorical, target = batch
        loss = self._loss(self(numeric, categorical), target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_index: int) -> None:
        numeric, categorical, target = batch
        logits = self(numeric, categorical)
        loss = self._loss(logits, target)
        probabilities = self._probabilities(logits)
        self.validation_ap.update(probabilities, target.long())
        self.validation_auc.update(probabilities, target.long())
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_ap", self.validation_ap, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.validation_auc, on_epoch=True)

    def test_step(self, batch: tuple[torch.Tensor, ...], batch_index: int) -> None:
        numeric, categorical, target = batch
        probabilities = self._probabilities(self(numeric, categorical))
        self.test_ap.update(probabilities, target.long())
        self.test_auc.update(probabilities, target.long())
        self.log("test_ap", self.test_ap, on_epoch=True)
        self.log("test_auc", self.test_auc, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
