import pandas as pd
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torch
import time
import os

def main():
    # Load processor and model
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", use_fast=False)
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    model = model.to(device)

    # Input and output paths
    input_file = "../hotpotQA_dataset_versions/11643paras_1200queries/English/11643_paras.csv"
    output_file = "../hotpotQA_dataset_versions/11643paras_1200queries/Urdu/11643_paras_translated.csv"

    # Remove old output file if exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Load data
    df = pd.read_csv(input_file)
    total_records = len(df)

    # Initialize timers and storage
    overall_start_time = time.time()
    batch_start_time = time.time()
    batch_original = []
    batch_translated = []

    for idx, content in enumerate(df['Content'], 1):
        record_start_time = time.time()

        # Prepare input
        text_inputs = processor(text=content, src_lang="eng", return_tensors="pt")
        text_inputs = {key: val.to(device) for key, val in text_inputs.items()}

        # Generate translation
        with torch.no_grad():
            output_tokens = model.generate(
                **text_inputs,
                tgt_lang="urd",
                generate_speech=False
            )
        translated_text = processor.decode(
            output_tokens[0].tolist()[0],
            skip_special_tokens=True
        )

        # Append batch
        batch_original.append(content)
        batch_translated.append(translated_text)

        record_time = time.time() - record_start_time
        print(f"✅ Translated record {idx}/{total_records} | Time: {record_time:.2f} seconds")

        # Write every 100 records
        if idx % 100 == 0 or idx == total_records:
            batch_df = pd.DataFrame({
                'Content': batch_original,
                'Translated_Content': batch_translated
            })
            write_header = not os.path.exists(output_file)
            batch_df.to_csv(output_file, mode='a', header=write_header, index=False, encoding="utf-8-sig")

            batch_time = time.time() - batch_start_time
            print(f"📦 Saved batch ending at record {idx} | Avg time per record: {batch_time / len(batch_df):.2f} sec\n")

            batch_start_time = time.time()
            batch_original = []
            batch_translated = []

        # Show total time every 1000 records
        if idx % 1000 == 0:
            elapsed = time.time() - overall_start_time
            print(f"⏳ Elapsed time after {idx} records: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)\n")

    # Final stats
    total_time = time.time() - overall_start_time
    avg_time = total_time / total_records

    print("\n🎉 All translations completed!")
    print(f"🕒 Total time: {total_time:.2f} seconds")
    print(f"⏱️ Average time per record: {avg_time:.2f} seconds")
    print(f"📄 Final CSV saved at: {output_file}")

if __name__ == "__main__":
    main()
