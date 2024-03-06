# import random
import torch
import torch.nn.functional as F


SYSTEM_PROMPT = ""
PROMPT_PREFIX = f"{SYSTEM_PROMPT}[|User|]"
PROMPT_SUFFIX = "</s>[|Assistant|]"


def create_batch_prompts(user_prompts):
    system_prompt = ""
    full_prompts = [
        f"{system_prompt}[|User|] {user_prompt.lower()}</s>[|Assistant|]"
        for user_prompt in user_prompts
    ]
    return full_prompts


def compute_num_audio_embeds(audio_samples, sr=16000):
    """
    Computes the number of audio embeddings that will be produced by the audio
    encoder. Note that the actual number may be off by one (less than the actual
    number). We assume that this will not affect the performance of the model.
    """
    # Pre-trained HuBERT produces embeddings every 20ms.
    num_embeds = (audio_samples - (sr * 0.01)) // (sr * 0.02)

    # Audio encoder further mean pools embeddings and downsamples by a factor of 4.
    num_pooled_embeds = int(num_embeds // 4 - 1)
    return num_pooled_embeds


def collate_audio_batch(data):
    # data contains 'audio', 'text', 'text_input_ids', 'llama2_response',
    # 'response_input_ids', and 'pool_ranges_4'
    # Collates only 'audio' in preparation for feeding into audio encoder.
    # text_input_ids and response_input_ids are left as as here.
    raw_audios = [x['audio']['array'] for x in data]
    audio_len_samples = [len(audio) for audio in raw_audios]
    max_len = max(audio_len_samples)
    prompt_texts = [x['text'] for x in data]
    ctc_pool_ranges = [x['pool_ranges_4'] for x in data]

    # Zero-pad audio on the right to match the longest audio clip in the batch.
    padded_audios = torch.stack(
        [F.pad(audio, (0, max_len - len(audio)), mode="constant") for audio in raw_audios],
        dim=0,
    ).float()

    # Return text_input_ids and response_input_ids as is without any padding;
    # these will be merged with the full sequence and collated later.
    text_input_ids = [x['text_input_ids'] for x in data]
    response_input_ids = [x['response_input_ids'] for x in data]

    return (
        padded_audios,
        audio_len_samples,
        prompt_texts,
        text_input_ids,
        response_input_ids,
        ctc_pool_ranges,
    )


def merge_prompt_response_tokens(
    prefix_input_ids, suffix_input_ids, inputs_embeds, response_input_ids, embed_tokens
):
    # Full embedding sequence is composed of prompt prefix (prefix_embeds),
    # prompt embeddings (inputs_embeds), prompt suffix (suffix_embeds),
    # and LLM response text embeddings (response_embeds).
    prefix_embeds = embed_tokens(prefix_input_ids)
    suffix_embeds = embed_tokens(suffix_input_ids)
    response_embeds = embed_tokens(response_input_ids)

    # Concatenate the components to make the full embedding sequence.
    full_embed_sequence = torch.cat(
        [
            prefix_embeds,
            inputs_embeds,
            suffix_embeds[:, 1:, :],
            response_embeds[:, 1:, :],
        ], dim=1
    )
    return full_embed_sequence


def merge_prompt_tokens(inputs_embeds, tokenizer, embed_tokens, device):
    # Concatenate the prompt prefix, prompt embeddings, and prompt suffix
    # in preparation for generation (LLM generates the response).
    prefix_input_ids = tokenizer(PROMPT_PREFIX, return_tensors="pt").input_ids.to(device)
    suffix_input_ids = tokenizer(PROMPT_SUFFIX, return_tensors="pt").input_ids.to(device)
    prefix_embeds = embed_tokens(prefix_input_ids)
    suffix_embeds = embed_tokens(suffix_input_ids)

    prompt_embed_sequence = torch.cat(
        [
            prefix_embeds,
            inputs_embeds,
            suffix_embeds[:, 1:, :],
        ], dim=1
    )
    return prompt_embed_sequence


def construct_attention_mask(seq_lens):
    max_len = max(seq_lens)
    unpadded_ones = [torch.ones(seq_len) for seq_len in seq_lens]
    attention_mask = torch.stack(
        [F.pad(ones, (max_len - len(ones), 0), mode="constant") for ones in unpadded_ones]
    ).long()
    return attention_mask


def batch_full_embed_sequence(
    all_audio_embeds,
    all_text_input_ids,
    all_response_input_ids,
    tokenizer,
    embed_tokens,
    device,
    process_text=False,
):
    # Get token IDs for instruction-tuned LLM's prompt prefix and suffix.
    prefix_input_ids = tokenizer(PROMPT_PREFIX, return_tensors="pt").input_ids.to(device)
    suffix_input_ids = tokenizer(PROMPT_SUFFIX, return_tensors="pt").input_ids.to(device)

    # Get embeddings for text prompt.
    unpadded_audio_embed_sequences = []
    unpadded_text_embed_sequences = []
    for audio_embeds, text_input_ids, response_input_ids in zip(
        all_audio_embeds, all_text_input_ids, all_response_input_ids
    ):
        # Get full audio prompt embedding sequence.
        full_audio_prompt_sequence = merge_prompt_response_tokens(
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            inputs_embeds=audio_embeds.unsqueeze(0),
            response_input_ids=response_input_ids.unsqueeze(0).to(device),
            embed_tokens=embed_tokens,
        )
        unpadded_audio_embed_sequences.append(full_audio_prompt_sequence)

        if process_text:
            # Get full text prompt embedding sequence.
            # HACK: Take elements [1:] to remove start of sentence token.
            text_embeds = embed_tokens(text_input_ids[1:].unsqueeze(0).to(device))
            full_text_prompt_sequence = merge_prompt_response_tokens(
                prefix_input_ids=prefix_input_ids,
                suffix_input_ids=suffix_input_ids,
                inputs_embeds=text_embeds,
                response_input_ids=response_input_ids.unsqueeze(0).to(device),
                embed_tokens=embed_tokens,
            )
            unpadded_text_embed_sequences.append(full_text_prompt_sequence)

    # Pad audio embedding sequences.
    audio_embed_sequence_lens = [seq.shape[1] for seq in unpadded_audio_embed_sequences]
    audio_max_len = max(audio_embed_sequence_lens)
    padded_audio_sequences = torch.cat(
        [
            F.pad(
                seq, (0, 0, audio_max_len - seq.shape[1], 0), mode="constant"
            ) for seq in unpadded_audio_embed_sequences
        ]
    )
    audio_attention_mask = construct_attention_mask(audio_embed_sequence_lens)

    if process_text:
        # Pad text embedding sequences.
        text_embed_sequence_lens = [seq.shape[1] for seq in unpadded_text_embed_sequences]
        text_max_len = max(text_embed_sequence_lens)
        padded_text_sequences = torch.cat(
            [
                F.pad(
                    seq, (0, 0, text_max_len - seq.shape[1], 0), mode="constant"
                ) for seq in unpadded_text_embed_sequences
            ]
        )
        text_attention_mask = construct_attention_mask(text_embed_sequence_lens)
    else:
        padded_text_sequences = None
        text_attention_mask = None

    return padded_audio_sequences, audio_attention_mask, padded_text_sequences, text_attention_mask


def soft_cross_entropy(input, target, reduction="mean"):
    s_likelihood = F.log_softmax(input, dim=-1)
    t_probability = F.softmax(target, dim=-1)
    cross_entropy = -torch.sum(t_probability * s_likelihood, dim=-1)
    if reduction == "mean":
        cross_entropy = cross_entropy.mean()
    else:
        pass
    return cross_entropy
