import argparse
import os
from omegaconf import OmegaConf
from tqdm import tqdm

import librosa
import torch
from datasets import load_from_disk

import datasets

from inference import LLMSpeechTextInference


class Evaluator():
    def __init__(self, llm_inferencer, eval_dataset, model_type):
        self.llm_inferencer = llm_inferencer
        self.eval_dataset = eval_dataset

        assert model_type in {"text", "audio", "cascade"}, "model_type must be one of 'text', 'audio', or 'cascade'."

        self.rouge = datasets.load_metric("rouge")
        self.meteor = datasets.load_metric("meteor")
        self.bertscore = datasets.load_metric("bertscore")

    def compute_ppl(self):
        pass

    def full_evaluate(self):
        for sample in self.eval_dataset:
            pass

        # metric = datasets.load_metric('my_metric')
        # for model_input, gold_references in evaluation_dataset:
        #     model_predictions = model(model_inputs)
        #     metric.add_batch(predictions=model_predictions, references=gold_references)
        # final_score = metric.compute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="yaml file for configuration")
    parser.add_argument('-g', '--gpu_idx', type=int, default=0,
                        help="index of home GPU device")
    parser.add_argument('-p', '--audio_encoder_checkpoint', type=str,
                        help="path to audio encoder checkpoint")
    args = parser.parse_args()

    # Select device for running models.
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")

    # Set up inferencer.
    config = OmegaConf.load(args.config)
    llm_inferencer = LLMSpeechTextInference(
        config=config,
        audio_encoder_checkpoint=args.audio_encoder_checkpoint,
        device=device,
    )
