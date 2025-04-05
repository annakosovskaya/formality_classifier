"""
Configuration parameters for the experiment
"""

# Model settings
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

# Evaluation settings
SHOT_COUNTS = [6, 5, 4, 3, 2, 1, 0]
TEST_SIZE = 3000
BATCH_SIZE = 14

# Data settings
DATA_PATH = "data/formality_dataset_multi.csv"

# Output settings
RESULTS_PLOT_PATH = "eval_results/few_shot_metrics_batched.png" 