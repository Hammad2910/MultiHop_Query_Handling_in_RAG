import pandas as pd
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torch
import time
import os

def translate_text(text, processor, model, device):
    inputs = processor(text=text, src_lang="eng", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            tgt_lang="urd",
            generate_speech=False
        )

    translated_text = processor.decode(
        output_tokens[0].tolist()[0],
        skip_special_tokens=True
    )
    return translated_text

def main():
    # Load model and processor
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", use_fast=False)
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model = model.to(device)

    # Paths
    input_path = "../hotpotQA_dataset_versions/5884paras_600queries/English/600_QnAs.csv"
    output_path = "../hotpotQA_dataset_versions/5884paras_600queries/Urdu/600_QnAs_translated.csv"

    # Load data
    df = pd.read_csv(input_path)
    total = len(df)

    # Prepare for processing
    translated_questions = []
    translated_answers = []
    translated_sentences = []
    start_time = time.time()
    batch_start = time.time()

    # Remove existing file if exists
    if os.path.exists(output_path):
        os.remove(output_path)

    for i, row in enumerate(df.itertuples(index=False), 1):
        q, a, s = row.question, row.answer, row.actual_retrieved_sentences

        row_start_time = time.time()
        try:
            translated_q = translate_text(q, processor, model, device)
            translated_a = translate_text(a, processor, model, device)
            translated_s = translate_text(s, processor, model, device)
        except Exception as e:
            print(f"⚠️ Error at row {i}: {e}")
            translated_q = translated_a = translated_s = ""

        row_end_time = time.time()
        row_time = row_end_time - row_start_time

        translated_questions.append(translated_q)
        translated_answers.append(translated_a)
        translated_sentences.append(translated_s)

        print(f"✅ Translated {i}/{total} - ⏱️ Time taken: {row_time:.2f} seconds")

        # Every 100 records or at the end
        if i % 100 == 0 or i == total:
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_size = len(translated_questions)
            avg_batch_time = batch_time / batch_size

            total_elapsed = batch_end - start_time
            avg_total_time = total_elapsed / i

            print(f"\n📦 Batch completed: {i - batch_size + 1} to {i}")
            print(f"⏱️ Avg time per record (last {batch_size}): {avg_batch_time:.2f} seconds")
            print(f"🕒 Total time elapsed: {total_elapsed:.2f} seconds")
            print(f"📊 Running avg time per record: {avg_total_time:.2f} seconds\n")

            # Write to file
            partial_df = df.iloc[i - batch_size:i].copy()
            partial_df['translated_question'] = translated_questions
            partial_df['translated_answer'] = translated_answers
            partial_df['translated_retrieved_sentences'] = translated_sentences

            header = not os.path.exists(output_path)
            partial_df.to_csv(output_path, mode='a', index=False, encoding='utf-8-sig', header=header)

            # Reset batch data
            translated_questions.clear()
            translated_answers.clear()
            translated_sentences.clear()
            batch_start = time.time()

    print(f"\n🎉 Translation completed and saved to: {output_path}")
    print(f"🕒 Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
