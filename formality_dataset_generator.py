import os
import pandas as pd
import re
import html
from datasets import load_dataset

# Set longer timeout for Hugging Face downloads
os.environ["HF_HUB_TIMEOUT"] = "60"

def clean_text(text):
    """Clean text by removing HTML entities, tags, and normalizing whitespace."""
    if not text:
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Trim excessive character repetition (e.g., "looooove" -> "loooove")
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_and_limit(texts, label, max_samples=200, min_len=20):
    """Process and limit examples with basic filtering."""
    filtered = [
        clean_text(t)
        for t in texts if t and len(t.strip()) >= min_len
    ]
    return pd.DataFrame({
        "text": filtered[:max_samples],
        "label": [label] * min(len(filtered), max_samples)
    })

def load_formal_sources(num_samples=200):
    """Load and process formal text sources."""
    print("Loading formal sources...")
    try:
        cnn = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{num_samples}]", trust_remote_code=True)
        xsum = load_dataset("xsum", split=f"train[:{num_samples}]", trust_remote_code=True)
        eurlex = load_dataset("lex_glue", "eurlex", split=f"train[:{num_samples}]", trust_remote_code=True)

        formal_df = pd.concat([
            clean_and_limit([x["article"] for x in cnn], "formal"),
            clean_and_limit([x["document"] for x in xsum], "formal"),
            clean_and_limit([x["text"] for x in eurlex], "formal"),
        ])
        return formal_df
    except Exception as e:
        print(f"Error loading formal sources: {str(e)}")
        raise

def load_informal_sources(num_samples=300):
    """Load and process informal text sources."""
    print("Loading informal sources...")
    try:
        tweets = load_dataset("tweet_eval", "emotion", split=f"train[:{num_samples}]", trust_remote_code=True)
        reddit = load_dataset("reddit", split=f"train[:{num_samples}]", trust_remote_code=True)
        ed = load_dataset("empathetic_dialogues", split=f"train[:{num_samples}]", trust_remote_code=True)

        informal_df = pd.concat([
            clean_and_limit([x["text"] for x in tweets], "informal"),
            clean_and_limit([x["body"] for x in reddit if "body" in x], "informal"),
            clean_and_limit([x["utterance"] for x in ed], "informal"),
        ])
        return informal_df
    except Exception as e:
        print(f"Error loading informal sources: {str(e)}")
        raise

def generate_formality_dataset(output_file="formality_dataset_multi.csv"):
    """Generate and save the complete formality dataset."""
    try:
        # Load both formal and informal sources
        formal_df = load_formal_sources()
        informal_df = load_informal_sources()

        # Combine and shuffle
        final_df = pd.concat([formal_df, informal_df]).sample(frac=1).reset_index(drop=True)

        # Save to CSV
        final_df.to_csv(output_file, index=False)
        
        print(f"Saved {len(final_df)} examples to {output_file}")
        print(f"There are {len(formal_df)} formal examples and {len(informal_df)} informal examples")
        
        return final_df

    except Exception as e:
        print(f"Error generating dataset: {str(e)}")
        raise

if __name__ == "__main__":
    generate_formality_dataset() 