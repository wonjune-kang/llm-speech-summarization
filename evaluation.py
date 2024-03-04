import argparse
import pickle
from omegaconf import OmegaConf
from tqdm import tqdm

import numpy as np
import torch

import evaluate
from datasets import load_from_disk

from inference import LLMSpeechTextInference
from utils import collate_audio_batch, batch_full_embed_sequence


def compute_nlls(inferencer, dataset, device, model_type):
    assert model_type in ["text", "audio", "cascade"]

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

                if model_type == "text" or model_type == "audio":
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

                    if model_type == "text":
                        prompt_embeds = full_text_prompt_sequence
                    else:
                        prompt_embeds = full_audio_prompt_sequence

                elif model_type == "cascade":
                    # Perform ASR with HuBERT.
                    asr_transcript = inferencer.perform_hubert_asr(audio)

                    # Tokenizer ASR transcript.
                    asr_transcript_input_ids = inferencer.llm_tokenizer(
                        asr_transcript,
                        return_tensors="pt",
                    ).input_ids

                    # Create full embedding sequence
                    (
                        _,
                        _,
                        prompt_embeds,
                        _,
                    ) = batch_full_embed_sequence(
                        all_audio_embeds=audio_embeds,
                        all_text_input_ids=asr_transcript_input_ids,
                        all_response_input_ids=response_input_ids,
                        tokenizer=inferencer.llm_tokenizer,
                        embed_tokens=inferencer.llm.model.embed_tokens,
                        device=device,
                        process_text=True,
                    )

                # Feed audio and text prompt sequences to LLM.
                llm_output = inferencer.llm(
                    inputs_embeds=prompt_embeds,
                    labels=response_input_ids[0].unsqueeze(0).to(device),
                )

                # Next token prediction losses for audio and text sequence inputs.
                ntp_loss = llm_output.loss

        # Compute perplexity from NLLs.
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
    print("\nEvaluating perplexity...")
    test_clean_nlls = compute_nlls(
        inferencer=llm_inferencer,
        dataset=librispeech_test_clean,
        device=device,
        model_type=args.model_type,
    )
    test_other_nlls = compute_nlls(
        inferencer=llm_inferencer,
        dataset=librispeech_test_other,
        device=device,
        model_type=args.model_type,
    )

    test_clean_ppl = torch.exp(torch.stack(test_clean_nlls).mean()).item()
    test_other_ppl = torch.exp(torch.stack(test_other_nlls).mean()).item()
    test_all_ppl = torch.exp(torch.stack(test_clean_nlls + test_other_nlls).mean()).item()

    print("\nPerplexity:")
    print("test-clean PPL:", round(test_clean_ppl, 4))
    print("test-other PPL:", round(test_other_ppl, 4))
    print("test-all PPL:", round(test_all_ppl, 4))
    print()

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
