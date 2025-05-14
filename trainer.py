import os
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

from model.audio_encoder import AudioEncoder
from model.audio_llama import AudioLlamaForCausalLM
from utils import (
    batch_full_embed_sequence,
    compute_num_audio_embeds,
    merge_prompt_tokens,
    soft_cross_entropy,
)
from writer import MyWriter


os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

        # Audio encoder.
        self.encoder_base = self.config.model.audio_encoder.base
        self.audio_encoder = AudioEncoder(self.config, self.device)
        print("Loaded audio encoder.\n")

        # LLM tokenizer and model.
        self.llm_type = self.config.model.llm_type
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_type,
            use_fast=False,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and freeze LLM model weights.
        self.llm = AudioLlamaForCausalLM.from_pretrained(
            self.llm_type,
            use_cache=True,
            torch_dtype=torch.float16,
        ).eval()
        for param in self.llm.parameters():
            param.requires_grad = False
        print("Loaded LLM.\n")

        # Set up train and validation dataloaders.
        self.get_dataloaders()
        print("Set up dataloaders.\n")

        # Flags for using logit and feature distillation losses.
        self.use_ld_loss = self.config.train.use_ld_loss
        self.use_fd_loss = self.config.train.use_fd_loss

        # Loss weighting.
        self.ntp_loss_weight = self.config.train.ntp_loss_weight
        self.ld_loss_weight = self.config.train.ld_loss_weight
        self.fd_loss_weight = self.config.train.fd_loss_weight

        # Connector layer indices for feature distillation loss.
        self.fd_loss_connector_layers = self.config.train.fd_loss_connector_layers

        # Send model to device.
        self.audio_encoder.to(self.device)
        self.llm.to(self.device)

        # Global training step and starting epoch for training run.
        self.step = 0
        self.start_epoch = 0

        # Gradient accumulation interval.
        self.grad_accum_interval = self.config.train.grad_accum_interval

        # Number of epochs to train.
        self.num_epochs = self.config.train.epochs

        # Set up optimizer and learning rate scheduler.
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.audio_encoder.parameters()},
                {'params': self.llm.parameters()}
            ],
            lr=self.config.train.optimizer.lr,
            betas=(self.config.train.optimizer.beta1, self.config.train.optimizer.beta2),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=(self.num_epochs * len(self.train_dataloader) // self.grad_accum_interval),
            power=1.0,
        )

        # Load checkpoint if specified.
        if self.args.checkpoint_path:
            self.load_checkpoint(self.args.checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.audio_encoder.load_state_dict(checkpoint["audio_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        # If training on GPU and loading optimizer state_dict, manually move
        # parameters to GPU.
        if self.device == torch.device(f"cuda:{self.args.gpu_idx}"):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(self.args.gpu_idx)

        print(f"Loaded checkpoint from {checkpoint_path}.\n")

    def collate_audio_batch_hubert(self, data):
        # data contains 'audio', 'text', 'text_input_ids', 'llm_response',
        # 'response_input_ids', and 'pool_ranges_4'
        # Collates only 'audio' in preparation for feeding into audio encoder.
        # text_input_ids and response_input_ids are left as as here.
        raw_audios = [x['audio']['array'] for x in data]
        audio_len_samples = [len(audio) for audio in raw_audios]
        max_len = max(audio_len_samples)
        prompt_texts = [x['text'] for x in data]
        ctc_pool_ranges = [x['pool_ranges_4'] for x in data]

        # Zero-pad audio on the right to match the longest audio clip in the batch.
        padded_audios = torch.stack(
            [F.pad(audio, (0, max_len - len(audio)), mode="constant") for audio in raw_audios],
            dim=0,
        ).float()

        # Return text_input_ids and response_input_ids as is without any padding;
        # these will be merged with the full sequence and collated later.
        # HACK: Take elements [1:] from input_ids to remove start of sequence
        # token that was included during data preprocessing.
        text_input_ids = [x['text_input_ids'][1:] for x in data]
        response_input_ids = [x['response_input_ids'][0][1:] for x in data]  # HACK/TODO: Remove [0]

        return (
            raw_audios,
            padded_audios,
            audio_len_samples,
            prompt_texts,
            text_input_ids,
            response_input_ids,
            ctc_pool_ranges,
        )

    def collate_audio_batch_whisper(self, data):
        # data contains 'audio', 'text', 'text_input_ids', 'llm_response',
        # 'response_input_ids', and 'pool_ranges_4'
        # Collates only 'audio' in preparation for feeding into audio encoder.
        # text_input_ids and response_input_ids are left as as here.
        raw_audios = [x['audio']['array'].numpy() for x in data]
        audio_len_samples = [len(audio) for audio in raw_audios]
        prompt_texts = [x['text'] for x in data]
        ctc_pool_ranges = [x['pool_ranges_4'] for x in data]

        padded_input_features = self.audio_encoder.feature_extractor(
            raw_audios,
            return_tensors="pt",
            sampling_rate=self.config.audio.sampling_rate,
        ).input_features

        # Return text_input_ids and response_input_ids as is without any padding;
        # these will be merged with the full sequence and collated later.
        # HACK: Take elements [1:] from input_ids to remove start of sequence
        # token that was included during data preprocessing.
        text_input_ids = [x['text_input_ids'][1:] for x in data]
        response_input_ids = [x['response_input_ids'][0][1:] for x in data]  # HACK/TODO: Remove [0]

        return (
            raw_audios,
            padded_input_features,
            audio_len_samples,
            prompt_texts,
            text_input_ids,
            response_input_ids,
            ctc_pool_ranges,
        )

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

        # # NOTE: For debugging only. Comment out below if not debugging.
        # self.train_dataset = self.train_dataset.select(range(50))
        # self.val_dataset = self.val_dataset.select(range(50))

        # Create dataloaders.
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=(
                self.collate_audio_batch_hubert
                if self.encoder_base == "hubert"
                else self.collate_audio_batch_whisper
            ),
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=(
                self.collate_audio_batch_hubert
                if self.encoder_base == "hubert"
                else self.collate_audio_batch_whisper
            ),
        )

    def train(self):
        # GradScaler for mixed precision training.
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            print(f"Epoch {epoch}")

            # Training loop.
            self.audio_encoder.train()
            self.optimizer.zero_grad()

            for batch_idx, (
                _,
                padded_inputs,
                audio_len_samples,
                _,
                text_input_ids,
                response_input_ids,
                ctc_pool_ranges,
            ) in enumerate(tqdm(self.train_dataloader)):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    padded_inputs = padded_inputs.to(self.device)
                    text_input_ids = [input_ids.to(self.device) for input_ids in text_input_ids]
                    response_input_ids = [
                        input_ids.to(self.device) for input_ids in response_input_ids
                    ]

                    # Compute audio embeddings using audio encoder.
                    padded_audio_embeds = self.audio_encoder(padded_inputs, ctc_pool_ranges)

                    if self.config.train.batch_size > 1 or self.encoder_base == "whisper":
                        # Unpad the audio embeddings in preparation for creating
                        # the full embedding sequence to feed into the LLM.
                        unpadded_audio_embeds = []
                        for padded_audio_embed, audio_samples in zip(
                            padded_audio_embeds, audio_len_samples
                        ):
                            num_audio_embeds = compute_num_audio_embeds(
                                audio_samples, sr=self.config.audio.sampling_rate
                            )
                            unpadded_audio_embed = padded_audio_embed[:num_audio_embeds, :]
                            unpadded_audio_embeds.append(unpadded_audio_embed)
                    else:
                        # If batch size = 1, no need to unpad by cropping.
                        unpadded_audio_embeds = padded_audio_embeds

                    # Create the full embedding sequence batch by concatenating
                    # the prompt prefix, audio embeddings, prompt suffix, and
                    # target LLM response.
                    (
                        batched_audio_prompt_sequences,
                        audio_attention_mask,
                        batched_text_prompt_sequences,
                        text_attention_mask,
                    ) = batch_full_embed_sequence(
                        all_audio_embeds=unpadded_audio_embeds,
                        all_text_input_ids=text_input_ids,
                        all_response_input_ids=response_input_ids,
                        tokenizer=self.tokenizer,
                        embed_tokens=self.llm.model.embed_tokens,
                        llm_type=self.llm_type,
                        device=self.device,
                        process_text=(self.use_ld_loss or self.use_fd_loss),
                    )

                    # Feed inputs_embeds to LLM.
                    audio_attention_mask = audio_attention_mask.to(self.device)
                    llm_audio_output = self.llm(
                        inputs_embeds=batched_audio_prompt_sequences,
                        labels=response_input_ids,
                        output_hidden_states=True,
                        attention_mask=audio_attention_mask,
                    )

                    # Keep track of total loss.
                    total_loss = 0.0
                    losses = {}

                    # Next token prediction loss from audio input.
                    ntp_loss = llm_audio_output.loss
                    total_loss += self.ntp_loss_weight * ntp_loss
                    losses["ntp_loss"] = ntp_loss.item()

                    if self.use_ld_loss or self.use_fd_loss:
                        num_labels = response_input_ids[0].shape[0]

                        # Perform forward pass with text inputs for distillation losses.
                        with torch.no_grad():
                            text_attention_mask = text_attention_mask.to(self.device)
                            llm_text_output = self.llm(
                                inputs_embeds=batched_text_prompt_sequences,
                                labels=response_input_ids,
                                output_hidden_states=True,
                                attention_mask=text_attention_mask,
                            )

                        # Logit distillation loss.
                        if self.use_ld_loss:
                            # NOTE: Assumes a batch size of 1.
                            ld_loss = soft_cross_entropy(
                                input=llm_audio_output.logits[:, -num_labels:, :],
                                target=llm_text_output.logits[:, -num_labels:, :].detach(),
                            )
                            total_loss += self.ld_loss_weight * ld_loss
                            losses["ld_loss"] = ld_loss.item()

                        # Feature distillation loss on LLM hidden states.
                        # NOTE: Assumes batch size = 1.
                        if self.use_fd_loss:
                            fd_loss = 0.0
                            for layer_idx in self.fd_loss_connector_layers:
                                audio_feats = llm_audio_output.hidden_states[layer_idx][
                                    :, -num_labels:, :
                                ]
                                text_feats = llm_text_output.hidden_states[layer_idx][
                                    :, -num_labels:, :
                                ]
                                fd_loss += F.mse_loss(audio_feats, text_feats.detach())

                            total_loss += self.fd_loss_weight * fd_loss
                            losses["fd_loss"] = fd_loss.item()

                # Normalize loss to account for gradient accumulation and do backward pass.
                total_loss /= self.grad_accum_interval
                scaler.scale(total_loss).backward()

                # Weights update.
                if (
                    ((batch_idx + 1) % self.grad_accum_interval == 0) or
                    (batch_idx + 1 == len(self.train_dataloader))
                ):
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                self.step += 1

                # Logging.
                if self.step % self.config.log.log_interval == 0:
                    self.writer.log_training(losses, self.step)
                    self.writer.log_lr(self.lr_scheduler.get_last_lr()[0], self.step)

                # Perform validation at interval.
                if self.step % self.config.log.validation_interval == 0:
                    self.validate(epoch)

            # Perform validation at end of epoch.
            self.validate(epoch)

    def validate(self, epoch):
        # Validation loop
        self.audio_encoder.eval()

        audio_nlls = []
        text_nlls = []
        prompt_audios = []
        prompt_texts = []
        llm_audio_responses = []
        llm_text_responses = []
        for sample_idx, (
            audio, input, _, texts, text_input_ids, response_input_ids, ctc_pool_ranges
        ) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    input = input.to(self.device)

                    # Compute audio embeddings using audio encoder.
                    audio_embeds = self.audio_encoder(input, ctc_pool_ranges)

                    # Create the full embedding sequence batch by concatenating
                    # the prompt prefix, audio embeddings, prompt suffix, and
                    # target LLM response.
                    (
                        full_audio_prompt_sequence,
                        _,
                        full_text_prompt_sequence,
                        _,
                    ) = batch_full_embed_sequence(
                        all_audio_embeds=audio_embeds,
                        all_text_input_ids=text_input_ids,
                        all_response_input_ids=response_input_ids,
                        tokenizer=self.tokenizer,
                        embed_tokens=self.llm.model.embed_tokens,
                        llm_type=self.llm_type,
                        device=self.device,
                        process_text=True,
                    )

                    # Feed audio and text prompt sequences to LLM.
                    llm_audio_output = self.llm(
                        inputs_embeds=full_audio_prompt_sequence,
                        labels=response_input_ids[0].unsqueeze(0).to(self.device),
                    )
                    llm_text_output = self.llm(
                        inputs_embeds=full_text_prompt_sequence,
                        labels=response_input_ids[0].unsqueeze(0).to(self.device),
                    )

                    # Next token prediction losses for audio and text sequence inputs.
                    audio_ntp_loss = llm_audio_output.loss
                    text_ntp_loss = llm_text_output.loss

                    # Perform generation using the audio and text prompts.
                    if sample_idx < self.config.log.num_generate_samples:
                        # Get prompt embedding sequences.
                        audio_prompt_emb_sequence = merge_prompt_tokens(
                            inputs_embeds=audio_embeds,
                            tokenizer=self.tokenizer,
                            embed_tokens=self.llm.model.embed_tokens,
                            llm_type=self.llm_type,
                            device=self.device,
                        )

                        text_embeds = self.llm.model.embed_tokens(
                            text_input_ids[0].unsqueeze(0).to(self.device)
                        )
                        text_prompt_emb_sequence = merge_prompt_tokens(
                            inputs_embeds=text_embeds,
                            tokenizer=self.tokenizer,
                            embed_tokens=self.llm.model.embed_tokens,
                            llm_type=self.llm_type,
                            device=self.device,
                        )

                        # Generate LLM responses to prompts.
                        audio_prompt_response = self.generate_llm_response(
                            inputs_embeds=audio_prompt_emb_sequence,
                            len_inputs=audio_embeds.shape[1],
                        )[0]
                        text_prompt_response = self.generate_llm_response(
                            inputs_embeds=text_prompt_emb_sequence,
                            len_inputs=audio_embeds.shape[1],  # Same len_inputs as audio.
                        )[0]

                        if isinstance(audio, torch.Tensor):
                            prompt_audios.append(audio.squeeze().cpu().numpy())
                        else:
                            prompt_audios.append(audio[0])
                        prompt_texts.append(texts[0])
                        llm_audio_responses.append(audio_prompt_response)
                        llm_text_responses.append(text_prompt_response)

            # Log loss in Tensorboard.
            losses = {"ntp_loss": audio_ntp_loss.item()}
            self.writer.log_validation(losses, self.step)

            # Compute perplexity from NLLs.
            audio_nlls.append(audio_ntp_loss)
            text_nlls.append(text_ntp_loss)

        # Log LLM responses in Tensorboard.
        self.writer.log_audio_text_responses(
            prompt_audios=prompt_audios,
            prompt_texts=prompt_texts,
            audio_responses=llm_audio_responses,
            text_responses=llm_text_responses,
            step=self.step,
        )

        # Log perplexity in Tensorboard.
        audio_perplexity = torch.exp(torch.stack(audio_nlls).mean())
        text_perplexity = torch.exp(torch.stack(text_nlls).mean())
        self.writer.log_validation_perplexity(audio_perplexity, "audio", self.step)
        self.writer.log_validation_perplexity(text_perplexity, "text", self.step)

        # Save checkpoints.
        save_path = os.path.join(self.checkpoint_save_dir, f"epoch_{epoch}_step_{self.step}.pt")
        torch.save(
            {
                "audio_encoder": self.audio_encoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "step": self.step,
            },
            save_path,
        )
        print(f"Saved checkpoint for epoch {epoch} to {save_path}.\n")

    def generate_llm_response(self, inputs_embeds, len_inputs=60):
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
