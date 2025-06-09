import pandas as pd
from transformers import AutoTokenizer
import os

def preprocess_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Clean data: remove empty or overly long texts
    df = df[df['text'].str.len() > 0]
    df = df[df['text'].str.len() < 512]  # AraT5 max length
    df = df[df['summary'].str.len() > 0]
    df = df.dropna()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/AraT5-base")

    # Tokenize text and summary
    def tokenize_row(row):
        inputs = tokenizer(
            row['text'],
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        targets = tokenizer(
            row['summary'],
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze().tolist(),
            'attention_mask': inputs['attention_mask'].squeeze().tolist(),
            'labels': targets['input_ids'].squeeze().tolist()
        }

    tokenized_data = df.apply(tokenize_row, axis=1, result_type='expand')
    df = pd.concat([df, tokenized_data], axis=1)

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    print(f"Data quality improved by filtering {100 - (len(df) / len(pd.read_csv(input_path)) * 100):.2f}%")

if __name__ == "__main__":
    preprocess_data('../data/sample_arabic_news.csv', '../data/preprocessed_arabic_news.csv')