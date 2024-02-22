import os
from tqdm.auto import tqdm

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
    merge_prompt_tokens,
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
        self.checkpoint_save_dir = os.path.join(self.config.log.checkpoint_dir, self.run_name)
        self.log_dir = os.path.join(self.config.log.log_dir, self.run_name)

        os.makedirs(self.checkpoint_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = MyWriter(self.config, self.log_dir)

        # Set up train and validation dataloaders.
        self.get_dataloaders()
        print("Set up dataloaders.\n")

        # Audio encoder.
        self.audio_encoder = AudioEncoder(self.config)
        print("Loaded audio encoder.\n")

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
        print("Loaded LLM.\n")

        # Send model to device.
        self.audio_encoder.to(self.device)
        self.llm.to(self.device)

        # Set up optimizer.
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.audio_encoder.parameters()},
                {'params': self.llm.parameters()}
            ],
            lr=self.config.train.optimizer.lr,
            betas=(self.config.train.optimizer.beta1, self.config.train.optimizer.beta2),
        )

        # Global training step.
        self.step = 0

        # Gradient accumulation interval.
        self.grad_accum_interval = self.config.train.grad_accum_interval

        # Number of epochs to train.
        self.num_epochs = self.config.train.epochs

        # Load checkpoint if specified.
        if self.args.checkpoint_path:
            self.load_checkpoint(self.args.checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
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

        # NOTE: For debugging only. Comment out if not debugging.
        self.train_dataset = self.train_dataset.select(range(500))
        self.val_dataset = self.val_dataset.select(range(500))

        # Create dataloaders.
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}")

            # Training loop.
            self.audio_encoder.train()
            self.optimizer.zero_grad()

            for batch_idx, (
                padded_audios, audio_len_samples, text_input_ids, response_input_ids
            ) in enumerate(tqdm(self.train_dataloader)):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    padded_audios = padded_audios.to(self.device)

                    # Compute audio embeddings using audio encoder.
                    unpadded_audio_embeds = []
                    padded_audio_embeds = self.audio_encoder(padded_audios)

                    # Unpad the audio embeddings in preparation for creating the
                    # full embedding sequence to feed into the LLM.
                    for padded_audio_embed, audio_samples in zip(
                        padded_audio_embeds, audio_len_samples
                    ):
                        num_audio_embeds = compute_num_audio_embeds(
                            audio_samples, sr=self.config.audio.sampling_rate
                        )
                        unpadded_audio_embed = padded_audio_embed[:num_audio_embeds, :]
                        unpadded_audio_embeds.append(unpadded_audio_embed)

                    # Create the full embedding sequence batch by concatenating
                    # the prompt prefix, audio embeddings, prompt suffix, and
                    # target LLM response.
                    batched_full_embed_sequence, attention_mask = batch_full_embed_sequence(
                        all_audio_embeds=unpadded_audio_embeds,
                        all_text_input_ids=None,
                        all_response_input_ids=response_input_ids,
                        tokenizer=self.tokenizer,
                        embed_tokens=self.llm.model.embed_tokens,
                        device=self.device,
                    )

                    # Feed inputs_embeds to LLM.
                    # TODO: Currently assumes batch size = 1 for labels -- need to change.
                    llm_audio_output = self.llm(
                        inputs_embeds=batched_full_embed_sequence,
                        labels=response_input_ids,
                        output_hidden_states=True,
                        attention_mask=attention_mask,
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

                    # Sum the two loss terms.
                    total_loss = ntp_loss  # + fd_loss

                # Normalize loss to account for gradient accumulation and do backward pass.
                total_loss /= self.grad_accum_interval
                total_loss.backward()

                # Weights update.
                if (
                    (self.step % self.grad_accum_interval == 0) or
                    (batch_idx + 1 == len(self.train_dataloader))
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.step += 1

                # Logging.
                if self.step % self.config.log.log_interval == 0:
                    losses = {
                        "ntp_loss": ntp_loss.item(),
                        # "fd_loss": fd_loss.item(),
                    }
                    self.writer.log_training(losses, self.step)

            # Validation loop
            self.audio_encoder.eval()

            prompt_audios = []
            prompt_texts = []
            llm_responses = []
            for sample_idx, (
                audio, audio_len_samples, text_input_ids, response_input_ids
            ) in enumerate(tqdm(self.val_dataloader)):
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        audio = audio.to(self.device)

                        # Compute audio embeddings using audio encoder.
                        audio_embeds = self.audio_encoder(audio)

                        # Create the full embedding sequence batch by concatenating
                        # the prompt prefix, audio embeddings, prompt suffix, and
                        # target LLM response.
                        batched_full_embed_sequence, _ = batch_full_embed_sequence(
                            all_audio_embeds=audio_embeds,
                            all_text_input_ids=None,  # TODO: Change for FD loss.
                            all_response_input_ids=response_input_ids,
                            tokenizer=self.tokenizer,
                            embed_tokens=self.llm.model.embed_tokens,
                            device=self.device,
                        )

                        # Feed inputs_embeds to LLM.
                        # TODO: Currently assumes batch size = 1 for labels -- need to change.
                        llm_audio_output = self.llm(
                            inputs_embeds=batched_full_embed_sequence,
                            labels=response_input_ids[0].unsqueeze(0).to(self.device),
                            output_hidden_states=True,
                        )

                        # Next token prediction loss from audio input.
                        ntp_loss = llm_audio_output.loss

                        if sample_idx < self.config.log.num_generate_samples:
                            # Get prompt embedding sequence.
                            prompt_emb_sequence = merge_prompt_tokens(
                                inputs_embeds=audio_embeds,
                                tokenizer=self.tokenizer,
                                embed_tokens=self.llm.model.embed_tokens,
                                device=self.device,
                            )

                            # Generate LLM response to audio prompt.
                            audio_prompt_response = self.generate_audio_prompt_response(
                                inputs_embeds=prompt_emb_sequence,
                                len_inputs=audio_embeds.shape[1],
                            )[0]

                            prompt_audios.append(audio.squeeze().cpu().numpy())
                            prompt_texts.append("Placeholder")
                            llm_responses.append(audio_prompt_response)

                    # Log loss in Tensorboard.
                    losses = {"ntp_loss": ntp_loss.item()}
                    self.writer.log_validation(losses, self.step)

            # Log LLM responses in Tensorboard.
            self.writer.log_audio_text_responses(
                prompt_audios=prompt_audios,
                prompt_texts=prompt_texts,
                llm_responses=llm_responses,
                epoch=epoch,
            )

            # Save checkpoints.
            save_path = os.path.join(self.checkpoint_save_dir, f"epoch_{epoch}.pt")
            torch.save(
                {
                    "audio_encoder": self.audio_encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "step": self.step,
                },
                save_path,
            )
            print(f"Saved checkpoint for epoch {epoch} to {save_path}.\n")

    def generate_audio_prompt_response(self, inputs_embeds, len_inputs=60):
        with torch.no_grad():
            # Generate
            generate_ids = self.llm.generate(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                max_new_tokens=2*len_inputs,
            )

        response_text = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return response_text
