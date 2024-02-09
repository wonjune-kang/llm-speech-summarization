from Typing import Dict, List, Optional, Tuple

import argparse
import os
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_from_disk
from transformers import LlamaForCausalLM, LlamaTokenizer

from model.audio_encoder import AudioEncoder
from model.llm_kd_loss import LLMKDLoss


def load_all_datasets(dataset_paths):
    all_datasets = []
    for split in dataset_paths:
        data_split = load_from_disk(split)
        all_datasets.append(data_split)
    return all_datasets


class Trainer():
    def __init__(self, args, config, device) -> None:
        self.args = args
        self.config = config

        self.run_name = args.run_name
        self.device = device

        self.checkpoint_path = os.path.join(self.config.train.checkpoint_path, self.run_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Audio encoder.
        self.audio_encoder = AudioEncoder()  # TODO

        # LLM tokenizer.
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "GeneZC/MiniChat-2-3B",
            use_fast=False,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and freeze LLM model weights.
        self.llm = LlamaForCausalLM.from_pretrained(
            "GeneZC/MiniChat-2-3B",
            use_cache=True,
            torch_dtype=torch.float16,
        )
        for param in self.llm.parameters():
            param.requires_grad = False
            param.grad = None

        # Set up optimizer.
        self.optimizer = torch.optim.AdamW(
            self.audio_encoder.parameters(),
            lr=self.config.train.optimizer.lr,
            betas=(self.config.train.optimizer.beta1, self.config.train.optimizer.beta2),
        )

        # Set up LLM KD loss function.
        self.kd_loss = LLMKDLoss()

        # Number of epochs to train.
        self.num_epochs = self.config.train.epochs

        # Keep track of loss values for checkpointing.
        self.best_epoch = -1
        self.best_avg_train_loss = np.inf
        self.best_avg_val_loss = np.inf

    def load_checkpoints(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self):
        pass
