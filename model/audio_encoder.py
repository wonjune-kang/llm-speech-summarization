import torch.nn as nn

from transformers import AutoModel


class AudioEncoder(nn.Module):
    """Audio Encoder network that converts """
    def __init__(self, config):
        super(AudioEncoder, self).__init__()
        self.config = config

        # Load pre-trained HuBERT model.
        self.encoder = AutoModel.from_pretrained(self.config.model.audio_encoder_type)

        self.pooling_layer = nn.AvgPool1d(
            kernel_size=self.config.model.pooling.kernel_size,
            stride=self.config.model.pooling.stride,
        )
        self.token_projection = nn.Linear(
            self.encoder.config.hidden_size,
            self.config.model.llm_embedding_channels,
        )

    def forward(self, audio_input):
        encoder_out = self.encoder(audio_input).last_hidden_state  # (B, N, 1024)
        pool_out = self.pooling_layer(
            encoder_out.transpose(1, 2)
        ).transpose(1, 2)
        audio_tokens = self.token_projection(pool_out)
        return audio_tokens
