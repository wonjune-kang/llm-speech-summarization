import os
from tqdm.auto import tqdm

import numpy as np
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import LlamaTokenizer

from model.audio_encoder import AudioEncoder
from model.audio_llama import AudioLlamaForCausalLM
# from model.feature_distillation_loss import FeatureDistillationLoss
from utils import (
    batch_full_embed_sequence,
    collate_audio_batch,
    compute_num_audio_embeds,
)
from writer import MyWriter


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

        # Set seed.
        torch.cuda.manual_seed(self.config.seed_everything)

        # Set up checkpointing and Tensorboard logging.
        self.checkpoint_path = os.path.join(self.config.log.checkpoint_path, self.run_name)
        self.log_dir = os.path.join(self.config.log.log_dir, self.run_name)

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = MyWriter(self.config, self.log_dir)

        # Audio encoder.
        self.audio_encoder = AudioEncoder(self.config)

        # LLM tokenizer.
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "GeneZC/MiniChat-2-3B",
            use_fast=False,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and freeze LLM model weights.
        self.llm = AudioLlamaForCausalLM.from_pretrained(
            "GeneZC/MiniChat-2-3B",
            use_cache=True,
            torch_dtype=torch.float16,
        ).eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        # Send model components to device.
        self.audio_encoder.to(self.device)
        self.llm.to(self.device)

        # Set up optimizer.
        self.optimizer = torch.optim.AdamW(
            self.audio_encoder.parameters(),
            lr=self.config.train.optimizer.lr,
            betas=(self.config.train.optimizer.beta1, self.config.train.optimizer.beta2),
        )

        # Global training step.
        self.step = 0

        # Number of epochs to train.
        self.num_epochs = self.config.train.epochs

        # Keep track of loss values for checkpointing.
        self.best_epoch = -1
        self.best_train_loss = np.inf
        self.best_val_loss = np.inf

        # Set up train and validation dataloaders.
        self.get_dataloaders()

    def load_checkpoints(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']

        # If training on GPU and loading optimizer state_dict, manually move
        # parameters to GPU.
        if self.device == torch.device(f"cuda:{self.args.gpu_idx}"):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(self.args.gpu_idx)

    def get_dataloaders(self):
        # Load train datasets and combine into one Dataset object.
        all_train_datasets = []
        for dataset_name in self.config.data.train_set:
            dataset_path = os.path.join(self.config.data.base_path, dataset_name)
            dataset = load_from_disk(dataset_path)
            dataset.set_format(type='torch')
            all_train_datasets.append(dataset)
        self.train_dataset = concatenate_datasets(all_train_datasets)

        # Load val datasets and combine into one Dataset object.
        all_val_datasets = []
        for dataset_name in self.config.data.val_set:
            dataset_path = os.path.join(self.config.data.base_path, dataset_name)
            dataset = load_from_disk(dataset_path)
            dataset.set_format(type='torch')
            all_val_datasets.append(dataset)
        self.val_dataset = concatenate_datasets(all_val_datasets)

        # Create dataloaders.
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            collate_fn=collate_audio_batch,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=collate_audio_batch,
        )

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}")

            self.audio_encoder.train()

            for (
                padded_audios, audio_len_samples, text_input_ids, response_input_ids
            ) in tqdm(self.train_dataloader):
                padded_audios = padded_audios.to(self.device)
                # text_input_ids = batch['input_ids'].to(self.device)
                # text_attention_mask = batch['attention_mask'].to(self.device)

                self.optimizer.zero_grad()

                # Compute audio embeddings using audio encoder.
                unpadded_audio_embeds = []
                padded_audio_embs = self.audio_encoder(padded_audios)

                # Unpad the audio embeddings in preparation for creating the
                # full embedding sequence to feed into the LLM.
                for padded_audio_embed, audio_samples in zip(padded_audio_embs, audio_len_samples):
                    num_audio_embeds = compute_num_audio_embeds(audio_samples)
                    unpadded_audio_embed = padded_audio_embed[:num_audio_embeds, :]
                    unpadded_audio_embeds.append(unpadded_audio_embed)

                # Create the full embedding sequence batch by concatenating
                # the prompt prefix, audio embeddings, prompt suffix, and
                # target LLM response.
                full_prompt_embed_sequence_batch = batch_full_embed_sequence(
                    all_audio_embeds=unpadded_audio_embeds,
                    all_text_input_ids=None,
                    all_response_input_ids=response_input_ids,
                )

                # Feed inputs_embeds to LLM.
                # TODO: Currently assumes batch size = 1 for labels -- need to change.
                llm_audio_output = self.llm(
                    inputs_embeds=full_prompt_embed_sequence_batch,
                    labels=response_input_ids[0].unsqueeze(0).to(self.device),
                    output_hidden_states=True,
                    # attention_mask=None,
                )

                # Next token prediction loss from audio input.
                ntp_loss = llm_audio_output.loss

                # # Feature distillation loss.
                # with torch.no_grad():
                #     llm_text_output = self.llm(
                #         input_ids=text_input_ids,
                #         attention_mask=text_attention_mask,
                #         labels=response_labels,
                #         output_hidden_states=True,
                #     )
                #     fd_loss = FeatureDistillationLoss(
                #         llm_audio_output.hidden_states,
                #         llm_text_output.hidden_states,
                #     )

                total_loss = ntp_loss  # + fd_loss

                total_loss.backward()
                self.optimizer.step()

                self.step += 1

                # Logging.
                if self.step % self.config.log.log_interval == 0:
                    losses = {
                        "ntp_loss": ntp_loss.item(),
                        # "fd_loss": fd_loss.item(),
                    }
                    self.writer.log_training(losses, self.step)

            # self.audio_encoder.eval()
            # for batch in tqdm(self.val_dataloader):
            #     pass

            # Save checkpoints.
            save_path = os.path.join(self.checkpoint_path, self.run_name, f"epoch_{epoch}.pt")
            torch.save(
                {
                    "audio_encoder": self.audio_encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "step": self.step,
                },
                save_path,
            )
            print(f"Saved checkpoint to {save_path}.\n")
