# evaluate.py

import os
import logging
import argparse

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, TrainingArguments

from utils import read_task_data, preprocess_function, compute_metrics

NUM_LABELS = 6

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_on_test_set(model_path, data_folder, dimension, max_length, batch_size, num_workers):
    """
    Evaluates the model on the test set.

    Args:
        model_path (str): Path to the trained model checkpoint.
        data_folder (str): Path to the data folder containing test.jsonl.
        dimension (str): The dimension to evaluate on (e.g., 'readability').
        max_length (int): The maximum sequence length.
        batch_size (int): The batch size for evaluation.
        num_workers (int): The number of workers for data processing.

    Returns:
        dict: A dictionary containing the evaluation metrics (accuracy, f1, precision, recall).
    """
    logging.info(f"Evaluating model from {model_path} on test set.")

    # 1. Read test data
    test_data = read_task_data(data_folder, dimension, choices=['test'])['test']
    test_dataset = Dataset.from_list(test_data)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length},
        num_proc=num_workers,
        load_from_cache_file=True
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir="./eval_results",  # Dummy directory, not used for saving
        per_device_eval_batch_size=batch_size,  # Set the batch size here
        report_to="none", # Disable reporting to avoid errors
        eval_strategy="no", # Disable evaluation during training
        save_strategy="no" # Disable saving during training
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        args=training_args
    )
    predictions = trainer.predict(tokenized_test_dataset)
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    return metrics


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a model on the test set.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder containing test.jsonl.')
    parser.add_argument('--dimension', type=str, default='readability', choices=['professionalism', 'readability', 'reasoning', 'cleanliness'], help='Dimension to evaluate on.')
    parser.add_argument('--max_length', type=int, default=8192, help='Maximum sequence length.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='Number of workers for data processing.')  # Default to CPU count
    return parser.parse_args()


def main():
    """Main function to parse arguments and run evaluation."""
    args = parse_args()
    metrics = evaluate_on_test_set(
        model_path=args.model_path,
        data_folder=args.data_folder,
        dimension=args.dimension,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()