import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    def __init__(self, llm, device):
        super(FeatureDistillationLoss, self).__init__()

        self.llm = llm
        self.device = device

    def forward(self, text_input_embeds, audio_input_embeds, labels):
        raise NotImplementedError()
