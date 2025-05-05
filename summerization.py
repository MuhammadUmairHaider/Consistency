import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import evaluate
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from rouge_score import rouge_scorer
from tqdm import tqdm

# Set up logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class CNNDailyMailDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=1024, split="train"):
        self.dataset = dataset[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Special tokens for separating article and summary
        # Add special tokens only once at the dataset initialization
        special_tokens = {'sep_token': '[SEP]'}
        if tokenizer.pad_token is None:
            special_tokens['pad_token'] = '[PAD]'
        
        # Add special tokens and resize vocabulary
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added_tokens} special tokens to the tokenizer")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        article = self.dataset[idx]["article"]
        highlights = self.dataset[idx]["highlights"]
        
        # Combine article and summary with separator
        text = f"{article} {self.tokenizer.sep_token} {highlights}"
        
        # Tokenize the combined text
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create attention mask and labels for language modeling
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        
        # Set labels equal to input_ids for causal language modeling
        labels = input_ids.clone()
        
        # Don't compute loss for padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_and_prepare_data(model_name="gpt2", max_length=1024):
    """Load CNN/Daily Mail dataset and prepare it for training"""
    # Load tokenizer 
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has the necessary tokens
    if tokenizer.pad_token is None:
        logger.info("Setting default pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset from Hugging Face
    logger.info("Loading CNN/Daily Mail dataset...")
    raw_datasets = load_dataset("cnn_dailymail", "3.0.0")
    
    # Initialize custom datasets
    train_dataset = CNNDailyMailDataset(raw_datasets, tokenizer, max_length, "train")
    val_dataset = CNNDailyMailDataset(raw_datasets, tokenizer, max_length, "validation")
    test_dataset = CNNDailyMailDataset(raw_datasets, tokenizer, max_length, "test")
    
    logger.info(f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test examples")
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    return tokenizer, train_dataset, val_dataset, test_dataset

def train_model(args):
    """Train the GPT model on CNN/Daily Mail summarization task"""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load and prepare datasets
    tokenizer, train_dataset, val_dataset, _ = load_and_prepare_data(
        model_name=args.model_name, 
        max_length=args.max_length
    )
    
    # Initialize model
    if args.from_scratch:
        logger.info(f"Training a new {args.model_name} model from scratch")
        model_config = GPT2Config.from_pretrained(
            args.model_name,
            vocab_size=len(tokenizer),
            n_positions=args.max_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = GPT2LMHeadModel(config=model_config)
    else:
        logger.info(f"Fine-tuning pre-trained {args.model_name} model")
        # First load the model with original config
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
        # Then resize token embeddings to match the new tokenizer size
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    
    # Set up data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model and tokenizer
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return model, tokenizer

def generate_summary(article_text, model, tokenizer, device, max_length=150):
    """Generate a summary for the given article text"""
    # Encode article text
    input_text = f"{article_text} {tokenizer.sep_token}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Generate summary
        output_ids = model.generate(
            inputs.input_ids,
            max_length=max_length + len(inputs.input_ids[0]),
            attention_mask=inputs.attention_mask,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode summary
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    
    # Extract the summary part (after the separator)
    sep_token = tokenizer.sep_token
    if sep_token in full_output:
        summary = full_output.split(sep_token)[1].strip()
        # Remove any special tokens from the summary
        summary = tokenizer.decode(tokenizer.encode(summary), skip_special_tokens=True)
    else:
        # If separator not found, take the last part of the output
        summary = " ".join(full_output.split()[-50:])
        # Remove any special tokens from the summary
        summary = tokenizer.decode(tokenizer.encode(summary), skip_special_tokens=True)
    
    return summary

def evaluate_model(model, tokenizer, test_dataset, args):
    """Evaluate the model using ROUGE metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Initialize ROUGE scorer
    rouge = evaluate.load("rouge")
    
    # Initialize results dictionary
    all_results = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "article_len": [],
        "ref_summary_len": [],
        "gen_summary_len": []
    }
    
    # Create test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    # Subset for evaluation if specified
    if args.eval_subset > 0:
        subset_size = min(args.eval_subset, len(test_dataloader))
        logger.info(f"Evaluating on a subset of {subset_size} examples")
        test_dataloader = list(test_dataloader)[:subset_size]
    
    logger.info("Generating summaries and computing metrics...")
    for batch in tqdm(test_dataloader):
        try:
            # Get article and reference summary by finding separator token
            input_ids = batch["input_ids"][0].tolist()
            
            # Find separator token index
            if tokenizer.sep_token_id in input_ids:
                article_idx = input_ids.index(tokenizer.sep_token_id)
            else:
                # If separator not found, skip this example
                logger.warning("Separator token not found in input, skipping example")
                continue
                
            article_ids = input_ids[:article_idx]
            reference_ids = input_ids[article_idx+1:]
            
            # Filter out padding tokens and -100 values (used for masking loss)
            article_ids = [id for id in article_ids if id != tokenizer.pad_token_id and id != -100]
            reference_ids = [id for id in reference_ids if id != tokenizer.pad_token_id and id != -100]
            
            # Decode article and reference summary
            article = tokenizer.decode(article_ids, skip_special_tokens=True)
            reference = tokenizer.decode(reference_ids, skip_special_tokens=True)
            
            # Skip if article or reference is empty
            if not article or not reference:
                logger.warning("Empty article or reference, skipping example")
                continue
            
            # Generate summary
            generated = generate_summary(article, model, tokenizer, device, max_length=args.summary_max_length)
            
            # Skip if generated summary is empty
            if not generated:
                logger.warning("Empty generated summary, skipping example")
                continue
            
            # Compute ROUGE scores
            rouge_result = rouge.compute(predictions=[generated], references=[reference])
            
            # Store results
            all_results["rouge1"].append(rouge_result["rouge1"])
            all_results["rouge2"].append(rouge_result["rouge2"])
            all_results["rougeL"].append(rouge_result["rougeL"])
            all_results["article_len"].append(len(article.split()))
            all_results["ref_summary_len"].append(len(reference.split()))
            all_results["gen_summary_len"].append(len(generated.split()))
            
        except Exception as e:
            logger.warning(f"Error processing example: {e}")
            continue
    
    # Check if we have any valid results
    if not all_results["rouge1"]:
        logger.error("No valid examples were processed during evaluation!")
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "avg_article_len": 0,
            "avg_ref_summary_len": 0,
            "avg_gen_summary_len": 0
        }, all_results
    
    # Calculate mean scores
    mean_results = {
        "rouge1": np.mean(all_results["rouge1"]),
        "rouge2": np.mean(all_results["rouge2"]),
        "rougeL": np.mean(all_results["rougeL"]),
        "avg_article_len": np.mean(all_results["article_len"]),
        "avg_ref_summary_len": np.mean(all_results["ref_summary_len"]),
        "avg_gen_summary_len": np.mean(all_results["gen_summary_len"])
    }
    
    return mean_results, all_results

def inference_loop(model_path, args):
    """Run inference loop on test set with appropriate metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    
    # Load test dataset
    _, _, _, test_dataset = load_and_prepare_data(model_name=model_path, max_length=args.max_length)
    
    # Evaluate model
    mean_results, all_results = evaluate_model(model, tokenizer, test_dataset, args)
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"ROUGE-1: {mean_results['rouge1']:.4f}")
    logger.info(f"ROUGE-2: {mean_results['rouge2']:.4f}")
    logger.info(f"ROUGE-L: {mean_results['rougeL']:.4f}")
    logger.info(f"Average Article Length: {mean_results['avg_article_len']:.1f} words")
    logger.info(f"Average Reference Summary Length: {mean_results['avg_ref_summary_len']:.1f} words")
    logger.info(f"Average Generated Summary Length: {mean_results['avg_gen_summary_len']:.1f} words")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        "rouge1": all_results["rouge1"],
        "rouge2": all_results["rouge2"],
        "rougeL": all_results["rougeL"],
        "article_len": all_results["article_len"],
        "ref_summary_len": all_results["ref_summary_len"],
        "gen_summary_len": all_results["gen_summary_len"]
    })
    
    results_df.to_csv(os.path.join(args.output_dir, "evaluation_results.csv"), index=False)
    
    # Example of interactive inference
    if args.interactive:
        logger.info("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            user_input = input("\nEnter an article to summarize (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
                
            summary = generate_summary(user_input, model, tokenizer, device, max_length=args.summary_max_length)
            print(f"\nGenerated Summary:\n{summary}")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a GPT model for CNN/Daily Mail summarization")
    
    # Model and data parameters
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the pre-trained model to use")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch instead of fine-tuning")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--summary_max_length", type=int, default=150, help="Maximum length for generated summaries")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    # Evaluation parameters
    parser.add_argument("--eval_subset", type=int, default=100, help="Number of examples to evaluate (0 for all)")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode after evaluation")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./cnn_dailymail_gpt", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Mode parameters
    parser.add_argument("--mode", type=str, choices=["train", "eval", "both"], default="both", 
                        help="Mode of operation: train, eval, or both")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to a trained model for evaluation (required for eval mode)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ["train", "both"]:
        # Train the model
        model, tokenizer = train_model(args)
        model_path = args.output_dir
    else:
        # Use provided model path for evaluation
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluation mode")
        model_path = args.model_path
    
    if args.mode in ["eval", "both"]:
        # Evaluate the model and run inference loop
        inference_loop(model_path, args)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()