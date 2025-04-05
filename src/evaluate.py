import torch
import pandas as pd
from tqdm.auto import tqdm
from src.prompts import create_zero_shot_prompt, create_few_shot_prompt
from src.predict_utils import extract_prediction

def evaluate_few_shot_batched(model, tokenizer2, data, max_seq_length, shot_counts=[0, 5, 10], test_size=100, batch_size=8):
    """
    Evaluate model performance with few-shot learning using batched processing.
    
    Args:
        model: The language model to use
        tokenizer2: Tokenizer for batch processing
        data: DataFrame containing 'text' and 'label' columns
        max_seq_length: Maximum sequence length for tokenizer
        shot_counts: List of number of shots to evaluate
        test_size: Number of test examples to use
        batch_size: Batch size for processing
    
    Returns:
        DataFrame with evaluation results
    """
    results = {}
    # tokenizer2 for batching 
    if tokenizer2.pad_token is None:
        print("Setting pad_token to eos_token for tokenizer2")
        if tokenizer2.eos_token:
             tokenizer2.pad_token = tokenizer2.eos_token # Important for batching
             print(f"Set tokenizer2.pad_token to: {tokenizer2.pad_token}")
        else:
             # Handle error: Cannot set pad_token if eos_token is also missing
             raise ValueError("tokenizer2 is missing eos_token, cannot set pad_token automatically.")


    for num_shots in shot_counts:
        print(f"\nEvaluating {num_shots}-shot classification with batch_size={batch_size}...")

        # Select examples for demonstration
        if num_shots > 0:
            examples = [(row['text'], row['label']) for _, row in data.head(num_shots).iterrows()]
            test_data_full = data.iloc[num_shots:].reset_index(drop=True)
        else:
            examples = []
            test_data_full = data

        # Limit test size
        test_data = test_data_full.head(test_size)

        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        # Process data in batches
        for i in tqdm(range(0, len(test_data), batch_size), desc=f"{num_shots}-shot Batches"):
            batch_df = test_data.iloc[i : i + batch_size]
            batch_texts = batch_df['text'].tolist()
            batch_true_labels = batch_df['label'].tolist()

            # Create prompts for the batch
            if num_shots > 0:
                batch_prompts = [create_few_shot_prompt(text, examples) for text in batch_texts]
            else:
                batch_prompts = [create_zero_shot_prompt(text) for text in batch_texts]

            # --- Batch Tokenization ---
            # Use tokenizer2 here
            inputs = tokenizer2(
                batch_prompts,
                return_tensors="pt",
                padding=True,         # Pad sequences to the longest in the batch
                truncation=True,
                max_length=max_seq_length - 15, # Reserve space for generation
                # padding_side should be implicitly handled by tokenizer2 setting
            ).to("cuda")

            # --- Batch Generation ---
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=15,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer2.pad_token_id
            )

            # --- Batch Decoding ---
            # Decode only the generated part (relative to the input length)
            batch_responses = []
            for idx in range(len(batch_prompts)):
                # Need input length based on padded input tokens from tokenizer2
                input_length = inputs["input_ids"][idx].shape[0]

                # Decode generated tokens for this specific example
                # Use tokenizer2 here
                generated_tokens = outputs[idx][input_length:]
                response = tokenizer2.decode(generated_tokens, skip_special_tokens=True).strip()
                batch_responses.append(response)


            # --- Process Batch Results ---
            for idx in range(len(batch_prompts)):
                prompt = batch_prompts[idx]
                response = batch_responses[idx]
                true_label = batch_true_labels[idx]

                prediction = extract_prediction(response, num_shots > 0)

                if prediction == true_label:
                    correct += 1

                total += 1
                all_predictions.append(prediction)
                all_labels.append(true_label)

                # Print first example of the first batch
                if i == 0 and idx == 0 and num_shots in [0, 5]: # Print example for 0 and 5 shots
                    print(f"\n💎 Example (Batch 1, Item 1):")
                    print(f"📜 Text: {batch_texts[idx][:200]}...") # Show truncated text
                    # print(f"➡️ Prompt: {prompt}")
                    print(f"✅ True label: {true_label}")
                    print(f"🗣️ Predicted: {prediction}")
                    print(f"🤖 Model output: {response}")

            # --- Clear Cache ---
            del inputs, outputs
            torch.cuda.empty_cache()
            # gc.collect()

        # --- Calculate Metrics for this num_shots ---
        if total == 0:
             print(f"{num_shots}-shot accuracy: N/A (No examples processed)")
             results[num_shots] = 0.0
             continue

        accuracy = correct / total
        print(f"{num_shots}-shot accuracy: {accuracy:.4f} ({correct}/{total})")

        formal_correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l and l == "formal")
        formal_total = sum(1 for l in all_labels if l == "formal")
        informal_correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l and l == "informal")
        informal_total = sum(1 for l in all_labels if l == "informal")
        formal_predicted = sum(1 for p in all_predictions if p == "formal")
        informal_predicted = sum(1 for p in all_predictions if p == "informal")

        if formal_total > 0:
            formal_recall = formal_correct / formal_total
            print(f"Formal recall: {formal_recall:.4f} ({formal_correct}/{formal_total})")
        else:
             print("Formal recall: N/A (0 examples)")
        if formal_predicted > 0:
            formal_precision = formal_correct / formal_predicted
            print(f"Formal precision: {formal_precision:.4f} ({formal_correct}/{formal_predicted})")
        else:
            print("Formal precision: N/A (0 predictions)")

        if formal_predicted > 0 and formal_total > 0:
            formal_f1 = 2 * (formal_precision * formal_recall) / (formal_precision + formal_recall)
            print(f"Formal F1: {formal_f1:.4f}")
        else:
            print("Formal F1: N/A (0 examples or 0 predictions)")

        if informal_total > 0:
            informal_recall = informal_correct / informal_total
            print(f"Informal recall: {informal_recall:.4f} ({informal_correct}/{informal_total})")
        else:
            print("Informal recall: N/A (0 examples)")

        if informal_predicted > 0:
            informal_precision = informal_correct / informal_predicted
            print(f"Informal precision: {informal_precision:.4f} ({informal_correct}/{informal_predicted})")
        else:
            print("Informal precision: N/A (0 predictions)")

        if informal_predicted > 0 and informal_total > 0:
            informal_f1 = 2 * (informal_precision * informal_recall) / (informal_precision + informal_recall)
            print(f"Informal F1: {informal_f1:.4f}")
        else:
            print("Informal F1: N/A (0 examples or 0 predictions)")

        # results[num_shots] = accuracy
        results[num_shots] = {
            "accuracy": accuracy,
            "formal_precision": formal_precision if formal_predicted > 0 else None,
            "formal_recall": formal_recall if formal_total > 0 else None,
            "formal_f1": formal_f1 if formal_predicted > 0 and formal_total > 0 else None,
            "informal_precision": informal_precision if informal_predicted > 0 else None,
            "informal_recall": informal_recall if informal_total > 0 else None,
            "informal_f1": informal_f1 if informal_predicted > 0 and informal_total > 0 else None,
        }

        # Clear memory before moving to next shot count
        torch.cuda.empty_cache()
        # gc.collect()

    results_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    results_df = results_df.rename(columns={"index": "num_shots"})

    return results_df