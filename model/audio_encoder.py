import torch
import torch.nn as nn

from transformers import HubertModel


class AudioEncoder(nn.Module):
    """Audio Encoder network that converts """
    def __init__(self, config=None):
        super(AudioEncoder, self).__init__()
        self.config = config

        # Load pre-trained HuBERT model.
        self.encoder = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

        self.pooling_layer = nn.AvgPool1d(
            kernel_size=self.config.model.pooling.kernel_size,
            stride=self.config.model.pooling.stride,
        )
        self.token_projection = nn.Linear(
            self.encoder.config.hidden_size,
            self.config.model.llm_embedding_channels,
        )

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(audio_input).last_hidden_state  # (B, N, 1024)
        pool_out = self.pooling_layer(
            encoder_out.transpose(1, 2)
        ).transpose(1, 2)
        audio_tokens = self.token_projection(pool_out)
        return audio_tokens
