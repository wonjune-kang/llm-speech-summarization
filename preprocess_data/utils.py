import random
import torch
from tqdm import tqdm


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_batch_prompts(user_prompts):
    system_prompt = ""
    full_prompts = [
        f"{system_prompt}[|User|] {user_prompt.lower()}</s>[|Assistant|]"
        for user_prompt in user_prompts
    ]
    return full_prompts


def run_llm_prompt_inference(user_prompt, llm, tokenizer, device):
    system_prompt = ""
    full_prompt = f"{system_prompt}[|User|] {user_prompt.lower()}</s>[|Assistant|]"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    full_prompt_embeds = llm.model.embed_tokens(inputs.input_ids)
    len_inputs = inputs.input_ids.shape[1]

    with torch.no_grad():
        # Generate
        seed_everything()
        generate_ids = llm.generate(
            input_ids=None,
            inputs_embeds=full_prompt_embeds,
            max_new_tokens=2*len_inputs,
        )

    outputs = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return outputs


def run_llm_prompt_inference_batched(user_prompts, llm, tokenizer, device):
    with torch.no_grad():
        full_prompts = create_batch_prompts(user_prompts)
        inputs = tokenizer(
            full_prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        full_prompt_embeds = llm.model.embed_tokens(inputs.input_ids)
        max_input_length = full_prompt_embeds.shape[1]

        # Generate
        seed_everything()
        generate_ids = llm.generate(
            input_ids=None,
            inputs_embeds=full_prompt_embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2*max_input_length,
        )

    outputs = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return outputs


def tokenize_and_clean_dataset(dataset, llm_tokenizer):
    dataset_with_tokenized_input_text = dataset.map(
        lambda x: llm_tokenizer(x['text'].lower())
    )
    dataset_with_tokenized_input_text = dataset_with_tokenized_input_text.rename_column(
        "input_ids", "text_input_ids"
    )

    dataset_with_tokenized_responses = dataset_with_tokenized_input_text.map(
        lambda x: llm_tokenizer(x['llm_response'])
    )
    dataset_with_tokenized_responses = dataset_with_tokenized_responses.rename_column(
        "input_ids", "response_input_ids"
    )

    dataset_with_tokenized_responses.set_format(
        columns=['audio', 'text', 'text_input_ids', 'llm_response', 'response_input_ids']
    )

    return dataset_with_tokenized_responses


def get_hubert_offsets(dataset, hubert, hubert_tokenizer, device, filter_by_length=True):
    # Filter samples that are greater than 20 seconds long.
    if filter_by_length:
        dataset = dataset.filter(lambda x: x['audio']['array'].shape[0]/16000 <= 20.0)

    word_offsets = []
    # char_offsets = []
    for sample in tqdm(dataset):
        audio = torch.tensor(sample['audio']['array']).unsqueeze(0).float()

        # forward sample through model to get greedily predicted transcription ids
        logits = hubert(audio.to(device)).logits[0]
        pred_ids = torch.argmax(logits, axis=-1)

        outputs = hubert_tokenizer.decode(
            pred_ids,
            output_word_offsets=True,
            # output_char_offsets=True,
        )

        word_offsets.append(outputs.word_offsets)
        # char_offsets.append(outputs.char_offsets)

    dataset_with_offsets = dataset.add_column("hubert_word_offsets", word_offsets)

    return dataset_with_offsets


def get_hubert_ctc_pool_ranges(dataset, pool_range=4):
    all_pool_ranges = []
    for sample in tqdm(dataset):
        hubert_word_offsets = sample["hubert_word_offsets"]
        ctc_word_offsets = [
            (word['start_offset'], word['end_offset']) for word in hubert_word_offsets
        ]

        all_word_offsets = [(0, 0, ctc_word_offsets[0][0])]
        for i in range(len(ctc_word_offsets)-1):
            all_word_offsets.append((1, ctc_word_offsets[i][0], ctc_word_offsets[i][1]))
            all_word_offsets.append((0, ctc_word_offsets[i][1], ctc_word_offsets[i+1][0]))
        all_word_offsets.append((1, ctc_word_offsets[-1][0], ctc_word_offsets[-1][1]))
        all_word_offsets.append(
            (0, ctc_word_offsets[-1][1], ctc_word_offsets[-1][1] + (pool_range * 2))
        )

        pool_ranges = []
        for is_word, start_offset, end_offset in all_word_offsets:
            if is_word == 1:
                startpoint = start_offset
                endpoint = start_offset + pool_range
                while startpoint < end_offset:
                    pool_ranges.append((startpoint, endpoint))
                    startpoint += pool_range
                    endpoint += pool_range
            else:
                pool_ranges.append((start_offset, end_offset))

        all_pool_ranges.append(pool_ranges)

    dataset_with_pool_ranges = dataset.add_column("pool_ranges_4", all_pool_ranges)

    return dataset_with_pool_ranges
