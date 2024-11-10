import argparse
import librosa
import torch
from omegaconf import OmegaConf
from transformers import LlamaTokenizer, AutoTokenizer, HubertForCTC

from model.audio_encoder import AudioEncoder
from model.audio_llama import AudioLlamaForCausalLM
from utils import merge_prompt_tokens, PROMPT_PREFIX, PROMPT_SUFFIX


class LLMSpeechTextInference():
    def __init__(self, config, audio_encoder_checkpoint, device):
        self.config = config
        self.device = device

        # Audio encoder.
        checkpoint = torch.load(audio_encoder_checkpoint, map_location="cpu")
        self.audio_encoder = AudioEncoder(self.config)
        self.audio_encoder.load_state_dict(checkpoint)
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

        # Load HuBERT ASR model for getting CTC offsets.
        self.hubert_tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
        self.hubert.to(self.device)
        print("Loaded HuBERT.\n")

    def perform_hubert_asr(self, audio):
        # Feed audio through model to get greedily predicted transcription IDs.
        logits = self.hubert(audio).logits[0]
        pred_ids = torch.argmax(logits, axis=-1)

        # Decode transcription IDs to get text transcript.
        # NOTE: Always converts to lower case.
        transcript = self.hubert_tokenizer.decode(pred_ids).lower()
        return transcript

    def get_ctc_pool_ranges(self, audio, pool_range=4):
        # Feed audio through model to get greedily predicted transcription IDs.
        logits = self.hubert(audio).logits[0]
        pred_ids = torch.argmax(logits, axis=-1)

        # Perform decoding to get CTC offsets for each predicted word.
        outputs = self.hubert_tokenizer.decode(pred_ids, output_word_offsets=True)
        word_offsets = outputs.word_offsets
        ctc_word_offsets = [
            (word['start_offset'], word['end_offset']) for word in word_offsets
        ]

        # Add offset ranges for silence in between words. The first element of
        # each tuple is a flag denoting whether the offset corresponds to
        # a word (1) or silence (0).
        all_word_offsets = [(0, 0, ctc_word_offsets[0][0])]
        for i in range(len(ctc_word_offsets)-1):
            all_word_offsets.append((1, ctc_word_offsets[i][0], ctc_word_offsets[i][1]))
            all_word_offsets.append((0, ctc_word_offsets[i][1], ctc_word_offsets[i+1][0]))
        all_word_offsets.append((1, ctc_word_offsets[-1][0], ctc_word_offsets[-1][1]))
        all_word_offsets.append(
            (0, ctc_word_offsets[-1][1], ctc_word_offsets[-1][1] + (pool_range * 2))
        )

        # Aggregate the offsets into pooling ranges for the audio encoder.
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

    def generate_llm_response(self, inputs_embeds, max_new_tokens=256):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # NOTE: Using greedy decoding for generation (no sampling).
                # Uncomment the lines below to change this.
                generate_ids = self.llm.generate(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    # do_sample=True,
                    # temperature=0.7,
                    max_new_tokens=max_new_tokens,
                )

        response_text = self.llm_tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return response_text

    def generate_text_response(self, input_text, max_new_tokens=256):
        # Create full prompt for instruction-tuned LLM.
        full_text_prompt = f"{PROMPT_PREFIX} {input_text}{PROMPT_SUFFIX} "

        with torch.no_grad():
            # Tokenize and get embeddings for the full text prompt.
            prompt_input_ids = self.llm_tokenizer(
                full_text_prompt, return_tensors='pt'
            ).input_ids.to(self.device)
            prompt_embeds = self.llm.model.embed_tokens(prompt_input_ids)

            # Generate the LLM response.
            llm_response = self.generate_llm_response(
                inputs_embeds=prompt_embeds,
                max_new_tokens=max_new_tokens,
            )[0]

        return llm_response

    def generate_asr_cascade_response(self, audio, additional_text_prompt="", max_new_tokens=256):
        with torch.no_grad():
            # Perform ASR using HuBERT.
            audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(self.device)
            asr_transcript = self.perform_hubert_asr(audio_tensor)

            # Combine the transcript with any additional text prompt.
            # NOTE: Assumes that the text prompt always comes before the
            # transcribed text.
            full_text = additional_text_prompt + asr_transcript
            llm_response = self.generate_text_response(full_text, max_new_tokens)

        return llm_response

    def generate_audio_response(self, audio, additional_text_prompt="", max_new_tokens=256):
        with torch.no_grad():
            audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(self.device)

            if self.audio_encoder.downsample_method == "ctc_pool":
                # Get the CTC pooling ranges for the audio.
                ctc_pool_ranges = self.get_ctc_pool_ranges(audio_tensor)

                # Get embeddings from the audio encoder.
                audio_embeds = self.audio_encoder(audio_tensor, [ctc_pool_ranges])
            else:
                audio_embeds = self.audio_encoder(audio_tensor, ctc_pool_ranges=None)

            # Combine the audio embeddings with any additional text prompt.
            # NOTE: Currently assumes that the text prompt always comes before
            # the audio. You can change how the embeddings are concatenated to
            # switch up the order or interleave text and audio prompts.
            if len(additional_text_prompt) > 0:
                # Take elements [1:] to remove start of sentence token.
                additional_text_input_ids = self.llm_tokenizer(
                    additional_text_prompt, return_tensors='pt'
                ).input_ids[:, 1:].to(self.device)

                # Get embeddings corresponding to additional text prompt and
                # concatenate with audio embeddings.
                text_embeds = self.llm.model.embed_tokens(additional_text_input_ids)
                combined_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
            else:
                # Otherwise, just use the audio embeddings.
                combined_embeds = audio_embeds

            # Get the full embedding sequence and generate the LLM response
            prompt_emb_sequence = merge_prompt_tokens(
                inputs_embeds=combined_embeds,
                tokenizer=self.llm_tokenizer,
                embed_tokens=self.llm.model.embed_tokens,
                device=self.device,
            )
            llm_response = self.generate_llm_response(prompt_emb_sequence, max_new_tokens)[0]

        return llm_response


if __name__ == '__main__':
    """
    Example use case for running generate_audio_response.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="yaml file for configuration")
    parser.add_argument('-g', '--gpu_idx', type=int, default=0,
                        help="index of home GPU device")
    parser.add_argument('-p', '--audio_encoder_checkpoint', type=str,
                        help="path to audio encoder checkpoint")
    parser.add_argument('-a', '--audio_file', type=str, required=True,
                        help="audio file containing speech utterance to be used in prompt")
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

    # Load audio file.
    audio, sr = librosa.load(args.audio_file, sr=16000)

    # Generate LLM response.
    # NOTE: Generating the response in this way sometimes leads to the LLM repeating a
    # chunk of text over and over. You can manually get around this by cropping the
    # generated output.
    llm_response = llm_inferencer.generate_audio_response(
        audio,
        max_new_tokens=512,
    )

    print("LLM Response:\n")
    print(llm_response)

