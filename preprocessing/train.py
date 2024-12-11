from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from accelerate import Accelerator
from transformers import get_scheduler
import nltk
from huggingface_hub import get_full_repo_name, Repository
from tqdm.auto import tqdm
import torch
import evaluate

# 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

# Initialize ROUGE evaluation metric
rouge_score = evaluate.load("rouge")

# Constants
MAX_INPUT_LEN = 1024
MAX_TARGET_LEN = 100
data_files = {"train": "train.csv", "test": "test.csv"}
checkpoint = "facebook/bart-large-cnn"

# Load the dataset
raw_data = load_dataset("csv", data_files=data_files)

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, quantization_config=BitsAndBytesConfig(load_in_4bit=True))


# Preprocessing function to tokenize the dataset
def preprocess_function(examples):
    inputs = tokenizer(
        examples["article"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )
    targets = tokenizer(
        examples["summary"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


# Tokenize the dataset
tokenized_dataset = raw_data.map(preprocess_function, batched=True)
tokenized_dataset.set_format("torch")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Initialize DataLoader
batch_size = 8
train_loader = DataLoader(
    tokenized_dataset["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
test_loader = DataLoader(
    tokenized_dataset["test"], collate_fn=data_collator, batch_size=batch_size
)

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Use Accelerate for distributed training
accelerator = Accelerator()
model, optimizer, train_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, test_loader
)

# Scheduler for learning rate decay
epochs = 10
update_steps_per_epoch = len(train_loader)
training_steps = epochs * update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_steps
)


# Postprocess function to handle the text formatting
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


# Initialize Hugging Face repo for saving model
model_name = f"{checkpoint.split('/')[-1]}-finetuned-news-summarizer"
repo_name = get_full_repo_name(model_name)
output_dir = "results-bart-cnn-finetuned"
repo = Repository(output_dir, clone_from=repo_name)

# Progress bar
progress_bar = tqdm(range(training_steps))

# Training loop
for epoch in range(epochs):
    # Training Phase
    model.train()
    for step, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation Phase
    model.eval()
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Post-process text to ensure ROUGE formatting
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            # Add to ROUGE metric batch
            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute metrics
    result = rouge_score.compute()
    # Extract and print ROUGE F1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    # Save and upload model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress, epoch {epoch}", blocking=False
        )
