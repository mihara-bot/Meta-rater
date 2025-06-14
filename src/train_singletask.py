import os
import argparse
from pathlib import Path
import logging

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

from utils import read_task_data, preprocess_function, compute_metrics, set_random_seed

NUM_LABELS = 6
DEFAULT_SEED = 42

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, help='pretrained model name or path')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--data_folder', type=str, help='data folder containing train.jsonl, val.jsonl, and test.jsonl')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max_length', type=int, default=8192)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dimension', type=str, default='readability', choices=['professionalism', 'readability', 'reasoning','cleanliness'])
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    args = parser.parse_args() 
    return args

def main():
    args = parse_args() 
    # Print arguments
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        logging.info("Arguments:")
        for k, v in sorted(vars(args).items()):
            logging.info(f"{k} = {v}")

    output_dir = Path(args.output_dir) 
    output_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    
    # 1.read data
    logging.info(f"Reading data from {args.data_folder}")
    data_dict = read_task_data(args.data_folder, args.dimension, choices=['train', 'val'])
    train_dataset, valid_dataset = Dataset.from_list(data_dict['train']), Dataset.from_list(data_dict['val']) 

    my_datasets = DatasetDict({
        'train':train_dataset,
        'valid':valid_dataset,
    })
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path) 
    tokenized_datasets = my_datasets.map(preprocess_function, 
                                         batched=True, 
                                         fn_kwargs={'tokenizer':tokenizer, 'max_length':args.max_length}, 
                                         num_proc=args.num_workers, 
                                         load_from_cache_file=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2.load model
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path, num_labels=NUM_LABELS)
    
    # 3.prepare training configs
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy='steps',
        evaluation_strategy='steps',
        save_steps=200,  # 64*200=12800 data
        eval_steps=200,
        learning_rate=args.learning_rate,
        lr_scheduler_type='cosine',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        seed=args.seed,
        report_to='none',   # disenable wandb
        save_total_limit=3,
        fp16=True,  # enable mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['valid'],
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'best_ckpt'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'best_ckpt'))

if __name__ == "__main__":
    main()