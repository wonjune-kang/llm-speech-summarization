import argparse

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf

from inference import LLMSpeechTextInference


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

    # Load CNN / DailyMail dataset for testing.
    cnn_dailymail = load_from_disk(
        "/u/wjkang/data/cnn_dailymail/cnn_dailymail_lt1600_with_audio.hf"
    )

    # Set up inferencer.
    config = OmegaConf.load(args.config)
    llm_inferencer = LLMSpeechTextInference(
        config=config,
        audio_encoder_checkpoint=args.audio_encoder_checkpoint,
        device=device,
    )

    for i in range(5):
        sample = cnn_dailymail[i]
        sample_text = sample["article"]
        sample_audio = sample["tts_audio"]

        text_prompt = "Summarize the following article in 4 sentences or less: "
        print("FULL TEXT PROMPT")
        print(text_prompt+sample_text)
        print()

        text_response = llm_inferencer.generate_text_response(text_prompt+sample_text)
        print("TEXT RESPONSE")
        print(text_response)
        print()

        cascade_response = llm_inferencer.generate_asr_cascade_response(
            audio=sample_audio,
            text_prompt=text_prompt,
        )
        print("CASCADE RESPONSE")
        print(cascade_response)
        print()

        audio_response = llm_inferencer.generate_audio_response(
            audio=sample_audio,
            text_prompt=text_prompt,
        )
        print("AUDIO RESPONSE")
        print(audio_response)
        print()
