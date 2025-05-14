import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel


def load_hubert_encoder(config):
    return AutoModel.from_pretrained(config.model.audio_encoder.type)


def load_whisper_encoder(config):
    encoder = AutoModel.from_pretrained(config.model.audio_encoder.type).encoder
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.model.audio_encoder.type)
    return encoder, feature_extractor


class AudioEncoder(nn.Module):
    def __init__(self, config, device):
        super(AudioEncoder, self).__init__()
        self.config = config
        self.device = device

        if self.config.model.audio_encoder.base == "hubert":
            self.encoder_base = "hubert"
            self.encoder = load_hubert_encoder(self.config)
        elif self.config.model.audio_encoder.base == "whisper":
            self.encoder_base = "whisper"
            self.encoder, self.feature_extractor = load_whisper_encoder(self.config)
        else:
            raise Exception("Unexpected encoder type in config.")

        self.downsample_method = self.config.model.audio_encoder.downsample_method
        self.downsample_factor = self.config.model.audio_encoder.downsample_factor

        if self.downsample_method == "pool":
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.config.model.audio_encoder.pooling.kernel_size,
                stride=self.config.model.audio_encoder.pooling.stride,
            )
            self.embed_projection = nn.Linear(
                self.encoder.config.hidden_size,
                self.config.model.llm_embedding_channels,
            )
        elif self.downsample_method == "stack":
            self.embed_projection = nn.Linear(
                self.encoder.config.hidden_size * self.downsample_factor,
                self.config.model.llm_embedding_channels,
            )
        elif self.downsample_method == "ctc_pool":
            self.embed_projection = nn.Linear(
                self.encoder.config.hidden_size,
                self.config.model.llm_embedding_channels,
            )
        else:
            raise Exception("Invalid downsampling method for audio encoder.")

    def forward(self, input, ctc_pool_ranges=None):
        encoder_out = self.encoder(input).last_hidden_state  # (B, N, 1024)

        if self.downsample_method == "pool":
            # (B, N, 1024) -> (B, N/4, 1024) -> (B, N/4, 3072)
            audio_embeds = self.pooling_layer(
                encoder_out.transpose(1, 2)
            ).transpose(1, 2)

        elif self.downsample_method == "stack":
            # (B, N, 1024) -> (B, N/4, 4096) -> (B, N/4, 3072)
            to_crop = encoder_out.shape[1] % self.downsample_factor
            audio_embeds = encoder_out[:, :-to_crop, :].reshape(
                1, -1, self.downsample_factor * encoder_out.shape[2]
            )

        elif self.downsample_method == "ctc_pool":
            assert ctc_pool_ranges is not None, (
                "Need to specify CTC pool ranges if using ctc_pool downsample method."
            )
            pooled_embs = []
            # NOTE: Assumes batch size = 1.
            for startpoint, endpoint in ctc_pool_ranges[0]:
                pooled_embs.append(
                    torch.mean(encoder_out[:, startpoint:endpoint, :], dim=1)
                )
            audio_embeds = torch.stack(pooled_embs, dim=1)

        else:
            raise Exception("Invalid downsampling method for audio encoder.")

        audio_embeds = self.embed_projection(audio_embeds)
        return audio_embeds
