# Prompting Large Language Models with Audio for General-Purpose Speech Summarization

### Wonjune Kang, Deb Roy

This repository contains code for training and running the audio encoder and LLM pipeline described in our Interspeech 2024 paper, Prompting Large Language Models with Audio for General-Purpose Speech Summarization, implemented in PyTorch. The system allows for the [MiniChat-3B LLM](https://huggingface.co/GeneZC/MiniChat-3B) to directly take in speech prompts as input instead of having to go through an intermediate automatic speech recognition (ASR) step. While we primarily used the system for speech summarization in the paper, it can be used for any general speech prompt as well.

Examples of the model's outputs (summaries and responses to general speech prompts) are available on our [demo page](https://llm-speech-summarization.github.io/).

If you find this work or our code useful, please consider citing our paper:

```
@article{kang2024prompting,
  title={Prompting Large Language Models with Audio for General-Purpose Speech Summarization},
  author={Kang, Wonjune and Roy, Deb},
  journal={arXiv preprint arXiv:2406.xxxxx},
  year={2024}
}
```

## Prerequisites

You can install all dependencies by running

```
pip install -r requirements.txt
```

## Pre-trained model weights

You can download the pretrained model checkpoint from the following Google Drive link:

**[Google Drive Link](https://drive.google.com/drive/folders/1o363nAqpyP80tivFNdjmyyoWGCLUeHZS?usp=sharing)**

## Data preprocessing

If you want to train a model from scratch, you will need to download the [Librispeech corpus](https://huggingface.co/datasets/librispeech_asr) from HuggingFace and preprocess the data.

**TODO**

## Training

You can train a model by specifying a config file (```-c```), GPU index (```-g```), and run name (```-n```). Training can also be continued from a checkpoint using the ```-p``` flag. **Note that the code currently only supports training on a single GPU with a batch size of 1.** This is because it was nontrivial to implement batching operations for the various loss computations (different amounts of padding needed for text and speech input lengths in knowledge distillation, different numbers of tokens in the ground-truth LLM responses, etc.).

```
RUN_NAME="full_model"
CONFIG_FILE="config/config_full.yaml"
GPU_IDX=0

python -u train.py -c $CONFIG_FILE -g $GPU_IDX -n $RUN_NAME
```

## Inference

You can perform inference using ```inference.py```. The script enables text response generation using MiniChat given:

1. A regular text prompt, using ```generate_text_response```
2. A speech prompt, using ```generate_audio_response```
3. A combination of both, using ```generate_audio_response``` while specifying ```additional_text_prompt```. For example, you could summarize a speech utterance by feeding it into ```audio``` and setting ```additional_text_prompt``` as the prompt for summarization (e.g., "Summarize the following article in 3 sentences or less: ")

## References

We referred to the following repositories and resources in our code:

- https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py for the core model structure of the LLM in ```model/audio_llama.py```
- https://github.com/GeneZC/MiniMA for the implementation of the soft cross-entropy in our logit distillation loss
- https://huggingface.co/GeneZC/MiniChat-3B for the pre-trained MiniChat-3B model
- https://huggingface.co/facebook/hubert-large-ls960-ft for the initial weights of the HuBERT-based audio encoder
