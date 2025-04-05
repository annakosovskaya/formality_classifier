import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import pandas as pd
from src.model_setup import setup_model
from src.evaluate import evaluate_few_shot_batched
from src.visualize import plot_and_print_results
from src.config import (
    MODEL_NAME, 
    MAX_SEQ_LENGTH, 
    LOAD_IN_4BIT,
    SHOT_COUNTS,
    TEST_SIZE,
    BATCH_SIZE,
    DATA_PATH,
    RESULTS_PLOT_PATH
)

# Setup model and tokenizers
model, _, tokenizer2 = setup_model(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT
)

# Load data and evaluate
data = pd.read_csv(DATA_PATH)
results_df = evaluate_few_shot_batched(
    model=model,
    tokenizer2=tokenizer2,
    data=data,
    max_seq_length=MAX_SEQ_LENGTH,
    shot_counts=SHOT_COUNTS,
    test_size=TEST_SIZE,
    batch_size=BATCH_SIZE
)

# Visualize results
plot_and_print_results(results_df, save_path=RESULTS_PLOT_PATH)

