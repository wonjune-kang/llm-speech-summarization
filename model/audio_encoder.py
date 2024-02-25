import torch.nn as nn
from transformers import AutoModel


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()
        self.config = config

        # Load pre-trained HuBERT model.
        self.encoder = AutoModel.from_pretrained(self.config.model.audio_encoder.type)

        self.downsample_method = self.config.model.audio_encoder.downsample_method
        self.downsample_factor = self.config.model.audio_encoder.downsample_factor

        if self.downsample_method == "pool":
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.config.model.audio_encoder.pooling.kernel_size,
                stride=self.config.model.audio_encoder.pooling.stride,
            )
            self.token_projection = nn.Linear(
                self.encoder.config.hidden_size,
                self.config.model.llm_embedding_channels,
            )
        elif self.downsample_method == "stack":
            self.token_projection = nn.Linear(
                self.encoder.config.hidden_size * self.downsample_factor,
                self.config.model.llm_embedding_channels,
            )
        else:
            raise Exception("Invalid downsampling method for audio encoder.")

    def forward(self, audio_input):
        encoder_out = self.encoder(audio_input).last_hidden_state  # (B, N, 1024)

        if self.downsample_method == "pool":
            audio_embeds = self.pooling_layer(
                encoder_out.transpose(1, 2)
            ).transpose(1, 2)
        elif self.downsample_method == "stack":
            to_crop = encoder_out.shape[1] % self.downsample_factor
            audio_embeds = encoder_out[:, :-to_crop, :].reshape(
                1, -1, self.downsample_factor * encoder_out.shape[2]
            )
        else:
            raise Exception("Invalid downsampling method for audio encoder.")

        audio_embeds = self.token_projection(audio_embeds)
        return audio_embeds

    def ctc_downsample(self, audio_input):
        # Returns a list of the number of embeddings to pool from the sequence.
        raise NotImplementedError()
