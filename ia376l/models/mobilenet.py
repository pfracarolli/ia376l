import torch
import pytorch_lightning as pl

from typing import Dict, Tuple, List, NewType
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.models import mobilenet_v3_small

TrainingStepOutput = NewType("TrainingStepOutput", Dict[str, float or torch.Tensor])


class MobileNet(pl.LightningModule):
    def __init__(self, criterion=CrossEntropyLoss()):
        super().__init__()

        self.net = mobilenet_v3_small(pretrained=False)
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> TrainingStepOutput:
        x, y = batch
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)

        accuracy = accuracy_score(
            y_pred.argmax(dim=-1).detach().cpu(), y.detach().cpu(), normalize=True
        )

        self.log("train_acc", accuracy, prog_bar=True)

        return {"loss": loss, "accuracy": accuracy}

    def validation_epoch_end(self, outputs: List[TrainingStepOutput]):
        mean_acc = 0

        for out in outputs:
            mean_acc += out["accuracy"]

        self.log("validation_acc", mean_acc / len(outputs))

    def test_epoch_end(self, outputs: List[TrainingStepOutput]):
        mean_acc = 0

        for out in outputs:
            mean_acc += out["accuracy"]

        self.log("test_acc", mean_acc / len(outputs))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-2)

    validation_step = training_step
    test_step = training_step
