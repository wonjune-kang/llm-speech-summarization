#!/usr/bin/python

import os
import torch

from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # HubertForCTC,
)

from utils import (
    run_llm_prompt_inference,
    # run_llm_prompt_inference_batched,
    tokenize_and_clean_dataset,
)


if __name__ == '__main__':
    gpu_idx = 0
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")

    # Initialize Llama 3.2 tokenizer and model.
    llm_tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        padding_size="left",
    )
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        use_cache=True,
        torch_dtype=torch.float16,
    ).eval().to(device)

    print("Loaded Llama 3.2 3B model.\n")

    # Load Librispeech-960h dataset.
    librispeech_all = load_dataset("librispeech_asr", "all")

    all_splits = [
        'train.clean.100',
        # 'train.clean.360',
        # 'train.other.500',
        'validation.clean',
        'validation.other',
        'test.clean',
        'test.other',
    ]

    hf_dataset_save_base_path = "/u/wjkang/data/librispeech_hf_llama3"
    for split in all_splits:
        # Preprocess each Librispeech split.
        librispeech_split = librispeech_all[split]

        print(f"Loaded Librispeech split {split}.\n")

        # STEP 1: Prompt LLM and get responses for each sample in Librispeech split.
        print("Generating LLM responses...")
        all_responses = []
        for sample in tqdm(librispeech_split):
            transcript = sample["text"].lower()
            response = run_llm_prompt_inference(
                user_prompt=transcript,
                model="llama3",
                llm=llm,
                tokenizer=llm_tokenizer,
                device=device,
            )
            all_responses.append(response)

        # Use batched inference to generate responses more quickly.
        # NOTE: The max_input_length variable in run_llm_prompt_inference_batched
        # technically makes the maximum decoding length twice the length of the
        # longest sample in the batch, not twice the length of the given sample.
        # This shouldn't affect training very much, but it is slightly different
        # from what was described in the paper. To stay true to the original
        # paper, uncomment and run the code segment above instead, but note that
        # this will slow things down quite a bit.
        # batch_size = 8
        # all_responses = []
        # for i in tqdm(range(0, len(librispeech_split), batch_size)):
        #     transcripts = [
        #         transcript.lower() for transcript in librispeech_split[i:i+batch_size]['text']
        #     ]
        #     responses = run_llm_prompt_inference_batched(
        #         user_prompts=transcripts,
        #         model="llama3",
        #         llm=llm,
        #         tokenizer=llm_tokenizer,
        #         device=device,
        #     )
        #     all_responses.extend(responses)

        assert len(all_responses) == len(librispeech_split), (
            "Should have generated the same number of responses as samples in the dataset!"
        )

        # Add responses to dataset.
        librispeech_split_with_responses = librispeech_split.add_column(
            "llm_response",
            all_responses,
        )

        # STEP 2: Pre-tokenize all input and response text in dataset.
        print("Tokenizing text transcripts and LLM responses...")
        librispeech_split_tokenized = tokenize_and_clean_dataset(
            librispeech_split_with_responses, llm_tokenizer
        )

        # Get dummy HuBERT word offsets and CTC pool ranges to keep dataset
        # format consistent with previous code.
        dummy_offsets = [[]] * len(librispeech_split_tokenized)
        librispeech_split_tokenized = librispeech_split_tokenized.add_column(
            "hubert_word_offsets", dummy_offsets
        )
        dummy_ctc_pool_ranges = [[]] * len(librispeech_split_tokenized)
        librispeech_split_tokenized = librispeech_split_tokenized.add_column(
            "pool_ranges_4", dummy_offsets
        )

        # Save dataset to disk.
        full_save_path = os.path.join(
            hf_dataset_save_base_path, f"librispeech_{split}_preprocessed.hf"
        )
        librispeech_split_tokenized.save_to_disk(full_save_path)
        print(f"Finished preprocessing {split} and saved dataset to disk at {full_save_path}.")
