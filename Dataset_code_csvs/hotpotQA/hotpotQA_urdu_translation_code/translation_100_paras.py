import pandas as pd
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torch
import time

def main():
    # Load processor and model
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", use_fast=False)
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    model.save_pretrained("C:/hammad workings/Thesis/dataset/translation task/translated datatset/translation_model")
    processor.save_pretrained("C:/hammad workings/Thesis/dataset/translation task/translated datatset/translation_model")

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load your CSV
    input_file = "C:/hammad workings/Thesis/dataset/paragraphs_for_100_records.csv"    # <<== Change this if needed
    df = pd.read_csv(input_file)

    # Create a list to store translated content
    translated_contents = []

    # Get total number of records
    total_records = len(df)

    # Start timing
    start_time = time.time()

    # Loop through each content
    for idx, content in enumerate(df['Content'], 1):
        # Record start time for one translation
        record_start_time = time.time()

        # Prepare input and move to device
        text_inputs = processor(
            text=content,
            src_lang="eng",
            return_tensors="pt"
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

        # Generate translation
        with torch.no_grad():
            output_tokens = model.generate(
                **text_inputs,
                tgt_lang="urd",
                generate_speech=False
            )

        # Decode
        translated_text = processor.decode(
            output_tokens[0].tolist()[0],
            skip_special_tokens=True
        )

        # Store
        translated_contents.append(translated_text)

        # Record end time for one translation
        record_end_time = time.time()
        record_time = record_end_time - record_start_time

        # Print progress
        print(f"✅ Translated record {idx}/{total_records} | Time: {record_time:.2f} seconds")

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / total_records

    # Add the new column
    df['Translated_Content'] = translated_contents

    # Save to a new CSV
    output_file = "C:/hammad workings/Thesis/dataset/translation task/translated datatset/translated_file_for_!00_paras.csv"
    df.to_csv(output_file, index=False)

    # Final summary
    print("\n🎉 All translations completed successfully!")
    print(f"🕒 Total time: {total_time:.2f} seconds")
    print(f"⏱️ Average time per record: {average_time:.2f} seconds")
    print(f"📄 Saved file as '{output_file}'")

if __name__ == "__main__":
    main()
