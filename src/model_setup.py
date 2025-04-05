import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from unsloth import tokenizer_utils

# needed as this function doesn't like it when the lm_head has its size changed
def do_nothing(*args, **kwargs):
    pass
tokenizer_utils.fix_untrained_tokens = do_nothing

def setup_model(
    model_name: str = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    dtype = None
):
    """
    Setup model and tokenizers for both regular and batch processing.
    
    Args:
        model_name: Name or path of the model to load
        max_seq_length: Maximum sequence length for the model
        load_in_4bit: Whether to load model in 4-bit precision
        dtype: Data type for model weights
        
    Returns:
        tuple: (model, tokenizer, tokenizer2)
    """
    # Regular tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        dtype=dtype,
    )

    # Batch processing tokenizer
    tokenizer2 = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
    )

    if tokenizer2.pad_token is None:
        if tokenizer2.eos_token:
            tokenizer2.pad_token = tokenizer2.eos_token
            print(f"Set tokenizer2.pad_token to: {tokenizer2.pad_token}")
        else:
            raise ValueError("tokenizer2 lacks both pad_token and eos_token.")

    FastLanguageModel.for_inference(model)

    return model, tokenizer, tokenizer2 
