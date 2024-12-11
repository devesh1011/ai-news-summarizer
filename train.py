from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_scheduler
import nltk
import torch
import evaluate
from peft import LoraConfig, get_peft_model

# from bitsandbytes import BitsAndBytesConfig

rouge_score = evaluate.load("rouge")

MAX_INPUT_LEN = 1024
MAX_TARGET_LEN = 100
data_files = {"train": "train.csv", "test": "test.csv"}
checkpoint = "facebook/bart-large-cnn"

raw_data = load_dataset("csv", data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
# Defined LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)


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


tokenized_dataset = raw_data.map(preprocess_function, batched=True)
tokenized_dataset.set_format("torch")

tokenized_dataset = tokenized_dataset.remove_columns(["article", "summary"])

batch_size = 8
train_loader = DataLoader(
    tokenized_dataset["train"],
    shuffle=True,
    # collate_fn=data_collator,
    batch_size=batch_size,
)
test_loader = DataLoader(tokenized_dataset["test"], batch_size=batch_size)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)

epochs = 3
update_steps_per_epoch = len(train_loader)
training_steps = epochs * update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_steps
)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


# Training loop
for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        # Move the batch data to the device
        batch = {key: value.to(device) for key, value in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(test_loader):
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            generated_tokens = torch.nn.functional.pad(
                generated_tokens,
                (0, MAX_TARGET_LEN - generated_tokens.shape[1]),
                value=tokenizer.pad_token_id,
            )
            labels = batch["labels"]

            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    result = rouge_score.compute()

    if isinstance(result, dict):
        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}

    print(f"Epoch {epoch}:", result)
