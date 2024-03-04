import argparse
from omegaconf import OmegaConf

import torch
from transformers import LlamaTokenizer, AutoTokenizer, HubertForCTC
from datasets import load_from_disk

from model.audio_encoder import AudioEncoder
from model.audio_llama import AudioLlamaForCausalLM
from utils import merge_prompt_tokens


class LLMSpeechTextInference():
    def __init__(self, config, audio_encoder_checkpoint, device):
        self.config = config
        self.device = device

        # Audio encoder.
        checkpoint = torch.load(audio_encoder_checkpoint, map_location="cpu")
        self.audio_encoder = AudioEncoder(self.config)
        self.audio_encoder.load_state_dict(checkpoint["audio_encoder"])
        self.audio_encoder.eval().to(self.device)
        print("Loaded audio encoder.\n")

        # LLM tokenizer.
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            "GeneZC/MiniChat-2-3B",
            use_fast=False,
        )

        # Load and freeze LLM model weights.
        self.llm = AudioLlamaForCausalLM.from_pretrained(
            "GeneZC/MiniChat-2-3B",
            use_cache=True,
            torch_dtype=torch.float16,
        ).eval()
        self.llm.to(self.device)
        print("Loaded LLM.\n")

        self.hubert_tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
        self.hubert.to(self.device)
        print("Loaded HuBERT.\n")

    def perform_hubert_asr(self, audio):
        # forward sample through model to get greedily predicted transcription ids
        logits = self.hubert(audio).logits[0]
        pred_ids = torch.argmax(logits, axis=-1)

        transcript = self.hubert_tokenizer.decode(pred_ids).lower()
        return transcript

    def get_ctc_pool_ranges(self, audio, pool_range=4):
        # forward sample through model to get greedily predicted transcription ids
        logits = self.hubert(audio).logits[0]
        pred_ids = torch.argmax(logits, axis=-1)

        outputs = self.hubert_tokenizer.decode(
            pred_ids,
            output_word_offsets=True,
            # output_char_offsets=True,
        )
        word_offsets = outputs.word_offsets

        ctc_word_offsets = [
            (word['start_offset'], word['end_offset']) for word in word_offsets
        ]

        all_word_offsets = [(0, 0, ctc_word_offsets[0][0])]
        for i in range(len(ctc_word_offsets)-1):
            all_word_offsets.append((1, ctc_word_offsets[i][0], ctc_word_offsets[i][1]))
            all_word_offsets.append((0, ctc_word_offsets[i][1], ctc_word_offsets[i+1][0]))

        all_word_offsets.append((1, ctc_word_offsets[-1][0], ctc_word_offsets[-1][1]))
        all_word_offsets.append(
            (0, ctc_word_offsets[-1][1], ctc_word_offsets[-1][1] + (pool_range * 2))
        )

        ctc_pool_ranges = []
        for is_word, start_offset, end_offset in all_word_offsets:
            if is_word == 1:
                startpoint = start_offset
                endpoint = start_offset + pool_range
                while startpoint < end_offset:
                    ctc_pool_ranges.append((startpoint, endpoint))
                    startpoint += pool_range
                    endpoint += pool_range
            else:
                ctc_pool_ranges.append((start_offset, end_offset))

        return ctc_pool_ranges

    def generate_llm_response(self, inputs_embeds):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Generate
                generate_ids = self.llm.generate(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    # do_sample=True,
                    # temperature=0.7,
                    max_new_tokens=256,
                )

        response_text = self.llm_tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return response_text

    def generate_text_response(self, text):
        with torch.no_grad():
            text_tokens = self.llm_tokenizer(text, return_tensors='pt').input_ids.to(self.device)
            text_embeds = self.llm.model.embed_tokens(text_tokens)
            text_prompt_emb_sequence = merge_prompt_tokens(
                inputs_embeds=text_embeds,
                tokenizer=self.llm_tokenizer,
                embed_tokens=self.llm.model.embed_tokens,
                device=self.device,
            )
            llm_response = self.generate_llm_response(text_prompt_emb_sequence)[0]

        if "\n" in llm_response:
            llm_response = llm_response.split("\n")[0]

        return llm_response

    def generate_asr_cascade_response(self, audio, text_prompt=""):
        with torch.no_grad():
            audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(self.device)
            asr_transcript = self.perform_hubert_asr(audio_tensor)
            llm_response = self.generate_text_response(text_prompt+asr_transcript)

        if "\n" in llm_response:
            llm_response = llm_response.split("\n")[0]

        return llm_response

    # def generate_audio_response(self, audio):
    #     audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(self.device)
    #     ctc_pool_ranges = self.get_ctc_pool_ranges(audio_tensor)
    #     audio_embeds = self.audio_encoder(audio_tensor, [ctc_pool_ranges])
    #     audio_prompt_emb_sequence = merge_prompt_tokens(
    #         inputs_embeds=audio_embeds,
    #         tokenizer=self.llm_tokenizer,
    #         embed_tokens=self.llm.model.embed_tokens,
    #         device=self.device,
    #     )
    #     llm_response = self.generate_llm_response(audio_prompt_emb_sequence)
    #     return llm_response

    def generate_audio_response(self, audio, text_prompt=""):
        with torch.no_grad():
            audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(self.device)
            ctc_pool_ranges = self.get_ctc_pool_ranges(audio_tensor)
            audio_embeds = self.audio_encoder(audio_tensor, [ctc_pool_ranges])

            if len(text_prompt) > 0:
                text_tokens = self.llm_tokenizer(
                    text_prompt, return_tensors='pt'
                ).input_ids.to(self.device)
                text_embeds = self.llm.model.embed_tokens(text_tokens)
                combined_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
            else:
                combined_embeds = audio_embeds

            prompt_emb_sequence = merge_prompt_tokens(
                inputs_embeds=combined_embeds,
                tokenizer=self.llm_tokenizer,
                embed_tokens=self.llm.model.embed_tokens,
                device=self.device,
            )
            llm_response = self.generate_llm_response(prompt_emb_sequence)[0]

        if "\n" in llm_response:
            llm_response = llm_response.split("\n")[0]

        return llm_response


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
