#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-05-22 14:50:34
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional
from loguru import logger

class Model(pl.LightningModule):
    def __init__(
        self, model_name: str, learning_rate: float = 1e-3, num_labels: int = 2
    ):
        super().__init__()
        logger.debug(f"Model name: {model_name}, learning_rate: {learning_rate}, num_labels: {num_labels}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.lr = learning_rate

    def forward(self, batch):

        results = self.model.forward(**batch)

        return results.logits

    def training_step(self, batch, batch_idx):

        inputs, labels = batch
        logits = self.forward(inputs)
        return {"loss": self.loss(logits, labels)}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss(logits, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def configure_optimizers(self):

        return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.lr)}


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        batch_size: int = 8,
        subtask: Optional[str] = None,
    ):
        super().__init__()
        self.dataset = (
            load_dataset(dataset_name, subtask)
            if subtask
            else load_dataset(dataset_name)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train = self.dataset["train"]
        self.val = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            [{"text": x["text"], "label": x["label"]} for x in self.train],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            [{"text": x["text"], "label": x["label"]} for x in self.val],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def collate_fn(self, examples):

        text = [e["text"] for e in examples]
        labels = [e["label"] for e in examples]

        results = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation="longest_first",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        return results, torch.from_numpy(np.asarray(labels)).long()


if __name__ == "__main__":

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(Model, DataModule)
