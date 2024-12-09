#!/usr/bin/python

import os
import torch

from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HubertForCTC,
    LlamaForCausalLM,
    LlamaTokenizer
)

from utils import (
    # run_llm_prompt_inference,
    run_llm_prompt_inference_batched,
    tokenize_and_clean_dataset,
    get_hubert_offsets,
    get_hubert_ctc_pool_ranges
)


if __name__ == '__main__':
    gpu_idx = 1
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")

    # Initialize MiniChat tokenizer and model.
    llm_tokenizer = LlamaTokenizer.from_pretrained(
        "GeneZC/MiniChat-2-3B",
        use_fast=False,
        padding_side="left",
    )
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    llm = LlamaForCausalLM.from_pretrained(
        "GeneZC/MiniChat-2-3B",
        use_cache=True,
        torch_dtype=torch.float16,
    ).eval().to(device)
    print("Loaded MiniChat LLM model.\n")

    # Load Librispeech-960h dataset.
    librispeech_all = load_dataset("librispeech_asr", "all")

    all_splits = [
        'train.clean.100',
        'train.clean.360',
        'train.other.500',
        'validation.clean',
        'validation.other',
        'test.clean',
        'test.other',
    ]

    hf_dataset_save_base_path = "/home/gridsan/wjkang/data/librispeech_hf"
    for split in all_splits:
        # Preprocess each Librispeech split.
        librispeech_split = librispeech_all[split]
        print(f"Loaded Librispeech split {split}.\n")

        # STEP 1: Prompt LLM and get responses for each sample in Librispeech split.
        print("Generating LLM responses...")
        # all_responses = []
        # for sample in tqdm(librispeech_split):
        #     transcript = sample["text"].lower()
        #     response = run_llm_prompt_inference(
        #         user_prompt=transcript,
        #         model="minichat",
        #         llm=llm,
        #         tokenizer=llm_tokenizer,
        #         device=device,
        #     )
        #     all_responses.append(response)

        # Use batched inference to generate responses more quickly.
        # NOTE: The max_input_length variable in run_llm_prompt_inference_batched
        # technically makes the maximum decoding length twice the length of the
        # longest sample in the batch, not twice the length of the given sample.
        # This shouldn't affect training very much, but it is slightly different
        # from what was described in the paper. To stay true to the original
        # paper, uncomment and run the code segment above instead, but note that
        # this will slow things down quite a bit.
        batch_size = 8
        all_responses = []
        for i in tqdm(range(0, len(librispeech_split), batch_size)):
            transcripts = [
                transcript.lower() for transcript in librispeech_split[i:i+batch_size]['text']
            ]
            responses = run_llm_prompt_inference_batched(
                user_prompts=transcripts,
                model="minichat",
                llm=llm,
                tokenizer=llm_tokenizer,
                device=device,
            )
            all_responses.extend(responses)

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

        # Load HuBERT.
        hubert = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
        hubert_tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
        print("Loaded HuBERT.")

        # STEP 3: Get HuBERT word offsets for computing CTC pool ranges.
        print("Computing HuBERT CTC word offsets...")
        librispeech_split_with_offsets = get_hubert_offsets(
            dataset=librispeech_split_tokenized,
            hubert=hubert,
            hubert_tokenizer=hubert_tokenizer,
            device=device,
        )

        # STEP 4: Get CTC pool ranges given HuBERT-predicted word offsets in
        # utterance. Only needed when using CTC offset-based pooling.
        print("Computing HuBERT CTC offset-based pool ranges...")
        librispeech_split_with_ctc_pool_ranges = get_hubert_ctc_pool_ranges(
            librispeech_split_with_offsets
        )

        # Save dataset to disk.
        full_save_path = os.path.join(
            hf_dataset_save_base_path, f"librispeech_{split}_preprocessed.hf"
        )
        librispeech_split_with_ctc_pool_ranges.save_to_disk(full_save_path)
        print(f"Finished preprocessing {split} and saved dataset to disk at {full_save_path}.")
