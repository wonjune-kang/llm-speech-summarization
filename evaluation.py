import argparse
import pickle
from omegaconf import OmegaConf
from tqdm import tqdm

import numpy as np
import torch

import evaluate
from datasets import load_from_disk

from inference import LLMSpeechTextInference
from utils import (
    batch_full_embed_sequence,
    collate_audio_batch,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
)


def collate_text_cascade_batch(data):
    # data contains 'audio', 'text', 'text_input_ids', 'llama2_response',
    # 'response_input_ids', and 'pool_ranges_4'
    raw_audios = [x['audio']['array'] for x in data]
    prompt_texts = [x['text'] for x in data]
    llm_responses = [x['llama2_response'] for x in data]
    response_input_ids = [x['response_input_ids'] for x in data]
    return raw_audios, prompt_texts, llm_responses, response_input_ids


def compute_text_nlls(inferencer, dataset, device):
    print("Text NLLs")
    test_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_text_cascade_batch,
    )

    nlls = []
    for _, prompt_texts, llm_responses, response_input_ids in tqdm(test_dataloader):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                text = prompt_texts[0].lower()
                response = llm_responses[0]
                response_ids = response_input_ids[0]

                text_seq = f"{PROMPT_PREFIX} {text}{PROMPT_SUFFIX} {response}"
                text_input_ids = inferencer.llm_tokenizer(
                    text_seq,
                    return_tensors="pt",
                ).input_ids

                llm_output = inferencer.llm(
                    input_ids=text_input_ids.to(device),
                    labels=response_ids.unsqueeze(0).to(device),
                )
                nlls.append(llm_output.loss)

    return nlls


def compute_cascade_nlls(inferencer, dataset, device):
    print("Cascade NLLs")
    test_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_text_cascade_batch,
    )

    nlls = []
    for audios, _, llm_responses, response_input_ids in tqdm(test_dataloader):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                audio = audios[0]
                response = llm_responses[0]
                response_ids = response_input_ids[0]

                # Perform ASR with HuBERT.
                asr_transcript = inferencer.perform_hubert_asr(audio.unsqueeze(0).to(device))

                text_seq = f"{PROMPT_PREFIX} {asr_transcript}{PROMPT_SUFFIX} {response}"
                text_input_ids = inferencer.llm_tokenizer(
                    text_seq,
                    return_tensors="pt",
                ).input_ids

                llm_output = inferencer.llm(
                    input_ids=text_input_ids.to(device),
                    labels=response_ids.unsqueeze(0).to(device),
                )
                nlls.append(llm_output.loss)

    return nlls


def compute_audio_nlls(inferencer, dataset, device):
    test_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_audio_batch,
    )

    nlls = []
    for sample_idx, (
        audio, _, texts, text_input_ids, response_input_ids, ctc_pool_ranges
    ) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                audio = audio.to(device)

                # Compute audio embeddings using audio encoder.
                audio_embeds = inferencer.audio_encoder(audio, ctc_pool_ranges)

                # Create the full embedding sequence batch by concatenating
                # the prompt prefix, audio embeddings, prompt suffix, and
                # target LLM response.
                (
                    full_audio_prompt_sequence,
                    _,
                    full_text_prompt_sequence,
                    _,
                ) = batch_full_embed_sequence(
                    all_audio_embeds=audio_embeds,
                    all_text_input_ids=text_input_ids,
                    all_response_input_ids=response_input_ids,
                    tokenizer=inferencer.llm_tokenizer,
                    embed_tokens=inferencer.llm.model.embed_tokens,
                    device=device,
                    process_text=True,
                )

                # Feed audio and text prompt sequences to LLM.
                llm_output = inferencer.llm(
                    inputs_embeds=full_audio_prompt_sequence,
                    labels=response_input_ids[0].unsqueeze(0).to(device),
                )

                # Next token prediction losses for audio and text sequence inputs.
                ntp_loss = llm_output.loss

        # Get NLLs to compute perplexity.
        nlls.append(ntp_loss)

    return nlls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="yaml file for configuration")
    parser.add_argument('-g', '--gpu_idx', type=int, default=0,
                        help="index of home GPU device")
    parser.add_argument('-p', '--audio_encoder_checkpoint', type=str,
                        help="path to audio encoder checkpoint")
    parser.add_argument('-t', '--model_type', type=str,
                        help="type of model to evaluate ('text', 'audio', or 'cascade')")
    args = parser.parse_args()

    # Select device for running models.
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")

    assert args.model_type in ["text", "audio", "cascade"]

    # Set up Librispeech test sets.
    librispeech_test_clean = load_from_disk(
        "/home/gridsan/wjkang/data/librispeech_hf/librispeech_test.clean_with_responses_tokenized_filtered_with_offsets_and_pool_ranges_4.hf"
    )
    librispeech_test_other = load_from_disk(
        "/home/gridsan/wjkang/data/librispeech_hf/librispeech_test.other_with_responses_tokenized_filtered_with_offsets_and_pool_ranges_4.hf"
    )
    librispeech_test_clean.set_format(type="torch")
    librispeech_test_other.set_format(type="torch")

    # Set up inferencer.
    config = OmegaConf.load(args.config)
    llm_inferencer = LLMSpeechTextInference(
        config=config,
        audio_encoder_checkpoint=args.audio_encoder_checkpoint,
        device=device,
    )

    # Compute perplexity on Librispeech.
    print("Evaluating perplexity...")

    if args.model_type == "text":
        test_clean_nlls = compute_text_nlls(
            inferencer=llm_inferencer,
            dataset=librispeech_test_clean,
            device=device,
        )
        test_other_nlls = compute_text_nlls(
            inferencer=llm_inferencer,
            dataset=librispeech_test_other,
            device=device,
        )
    elif args.model_type == "audio":
        test_clean_nlls = compute_audio_nlls(
            inferencer=llm_inferencer,
            dataset=librispeech_test_clean,
            device=device,
        )
        test_other_nlls = compute_audio_nlls(
            inferencer=llm_inferencer,
            dataset=librispeech_test_other,
            device=device,
        )
    elif args.model_type == "cascade":
        test_clean_nlls = compute_cascade_nlls(
            inferencer=llm_inferencer,
            dataset=librispeech_test_clean,
            device=device,
        )
        test_other_nlls = compute_cascade_nlls(
            inferencer=llm_inferencer,
            dataset=librispeech_test_other,
            device=device,
        )

    test_clean_ppl = torch.exp(torch.stack(test_clean_nlls).mean()).item()
    test_other_ppl = torch.exp(torch.stack(test_other_nlls).mean()).item()
    test_all_ppl = torch.exp(torch.stack(test_clean_nlls + test_other_nlls).mean()).item()

    print("\nPerplexity:")
    print("test-clean PPL:", round(test_clean_ppl, 4))
    print("test-other PPL:", round(test_other_ppl, 4))
    print("test-all PPL:", round(test_all_ppl, 4))
    print()

    exit()

    # Perform summarization on CNN / DailyMail articles.
    cnn_dailymail = load_from_disk(
        "/home/gridsan/wjkang/data/cnn_dailymail/cnn_dailymail_lt1600_with_audio.hf"
    )

    print("\nEvaluating summarization...")
    all_summaries = []
    all_gt_summaries = []
    all_sample_ids = []
    for sample in tqdm(cnn_dailymail):
        sample_id = sample["id"]
        sample_text = sample["article"]
        sample_gt_summary = sample["highlights"]
        sample_audio = sample["tts_audio"]

        all_sample_ids.append(sample_id)
        all_gt_summaries.append(sample_gt_summary)

        # Prompt for performing summarization.
        text_prompt = "Summarize the following article in 3 sentences or less: "

        # Prompt the LLM to perform summarization.
        if args.model_type == "text":
            llm_summary = llm_inferencer.generate_text_response(text_prompt+sample_text)
        elif args.model_type == "cascade":
            llm_summary = llm_inferencer.generate_asr_cascade_response(
                audio=sample_audio,
                text_prompt=text_prompt,
            )
        elif args.model_type == "audio":
            llm_summary = llm_inferencer.generate_audio_response(
                audio=sample_audio,
                text_prompt=text_prompt,
            )
        all_summaries.append(llm_summary)

        print("ARTICLE")
        print(sample_text)
        print()
        print("GT SUMMARY")
        print(sample_gt_summary)
        print()
        print("LLM SUMMARY")
        print(llm_summary)

        if len(all_summaries) == 5:
            break

    # Save inference outputs for faster evaluation next time.
    save_info = {}
    for sample_id, llm_summary, gt_summary in zip(
        all_sample_ids, all_summaries, all_gt_summaries
    ):
        save_info[sample_id] = {
            "gt_summary": gt_summary,
            "llm_summary": llm_summary,
        }

    if args.model_type == "text":
        save_filename = "text_summaries.pkl"
    elif args.model_type == "cascade":
        save_filename = "cascade_summaries.pkl"
    else:
        save_filename = f"{args.audio_encoder_checkpoint[:-3]}_summaries.pkl"

    with open(save_filename, 'wb') as f:
        pickle.dump(save_info, f)
    print(f"Saved computed summaries to file {save_filename}.")

    # # Set up evaluation metrics.
    # rouge_metric = evaluate.load("rouge")
    # meteor_metric = evaluate.load("meteor")
    # bertscore_metric = evaluate.load("bertscore")

    # rouge = rouge_metric.compute(predictions=all_summaries, references=all_gt_summaries)
    # meteor = meteor_metric.compute(predictions=all_summaries, references=all_gt_summaries)
    # bertscore = bertscore_metric.compute(
    #     predictions=all_summaries,
    #     references=all_gt_summaries,
    #     lang="en",
    #     device=device,
    # )

    # print("\nSummarization metrics:")

    # print("ROUGE")
    # for metric, value in rouge.items():
    #     print(metric, round(value, 4))
    # print()

    # print("METEOR")
    # for metric, value in meteor.items():
    #     print(metric, round(value, 4))
    # print()

    # print("BERTScore")
    # print("Precision:", round(np.mean(bertscore["precision"]), 4))
    # print("Recall:", round(np.mean(bertscore["recall"]), 4))
    # print("F1:", round(np.mean(bertscore["f1"]), 4))
    # print()
