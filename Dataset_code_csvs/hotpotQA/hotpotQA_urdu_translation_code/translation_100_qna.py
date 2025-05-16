import pandas as pd
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torch
import time

def translate_text(text, processor, model, device):
    # Tokenize and move to device
    inputs = processor(text=text, src_lang="eng", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate translation
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            tgt_lang="urd",
            generate_speech=False
        )
    
    # Decode
    translated_text = processor.decode(
        output_tokens[0].tolist()[0],
        skip_special_tokens=True
    )
    return translated_text

def main():
    # Load model and processor
    processor = AutoProcessor.from_pretrained("C:/hammad workings/Thesis/hotpotqa_translation_task/translation_task_storage/dataset and weights/translation_model_weights", use_fast=False)
    model = SeamlessM4Tv2Model.from_pretrained("C:/hammad workings/Thesis/hotpotqa_translation_task/translation_task_storage/dataset and weights/translation_model_weights")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # File paths
    input_path = "C:/hammad workings/Thesis/hotpotqa_translation_task/dataset_100_retrived_sen.csv"
    output_path = "C:/hammad workings/Thesis/hotpotqa_translation_task/translation_task_storage/dataset and weights/translated_dataset_100_qna.csv"

    # Load dataset
    df = pd.read_csv(input_path)

    # Initialize lists for translations
    translated_questions = []
    translated_answers = []
    translated_sentences = []

    total = len(df)
    start_time = time.time()

    for i, row in enumerate(df.itertuples(index=False), 1):
        q = row.question
        a = row.answer
        s = row.actual_retrieved_sentences
        try:
            translated_q = translate_text(q, processor, model, device)
            translated_a = translate_text(a, processor, model, device)
            translated_s = translate_text(s, processor, model, device)
        except Exception as e:
            print(f"⚠️ Error at row {i}: {e}")
            translated_q = translated_a = translated_s = ""

        translated_questions.append(translated_q)
        translated_answers.append(translated_a)
        translated_sentences.append(translated_s)

        print(f"✅ Translated {i}/{total}")

    # Add translations to DataFrame
    df['translated_question'] = translated_questions
    df['translated_answer'] = translated_answers
    df['translated_retrieved_sentences'] = translated_sentences

    # Save result
    df.to_csv(output_path, index=False,encoding="utf-8-sig")
    print(f"\n🎉 Translation completed and saved to: {output_path}")
    print(f"🕒 Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
