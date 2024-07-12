import os
import requests
import zipfile
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AdamW
import torch
import time
from torch.utils.data import Dataset

# Function to download the dataset with retry mechanism
def download_dataset(url, dest_path, retries=5, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            with open(dest_path, "wb") as f:
                f.write(response.content)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return False

# Download the dataset
dataset_url = "http://yanran.li/files/ijcnlp_dailydialog.zip"
dataset_path = "ijcnlp_dailydialog.zip"
if not download_dataset(dataset_url, dataset_path):
    print("Failed to download the dataset. Please try again later or download it manually.")
    exit(1)

# Unzip the main dataset file
with zipfile.ZipFile(dataset_path, "r") as zip_ref:
    zip_ref.extractall("dailydialog")

# Check and unzip inner zip files if they exist
inner_dir = "dailydialog/ijcnlp_dailydialog"
if not os.path.exists(inner_dir):
    print(f"Expected directory {inner_dir} not found.")
    exit(1)

# Unzip the train, validation, and test sets
for split in ['train', 'validation', 'test']:
    split_zip_path = os.path.join(inner_dir, f"{split}.zip")
    split_extract_path = os.path.join(inner_dir, split)
    if os.path.exists(split_zip_path):
        with zipfile.ZipFile(split_zip_path, "r") as zip_ref:
            zip_ref.extractall(split_extract_path)
    else:
        print(f"Expected file {split_zip_path} not found.")
        exit(1)

# Load and preprocess the data
train_file_path = os.path.join(inner_dir, "train/train/dialogues_train.txt")
if not os.path.exists(train_file_path):
    print(f"Expected file {train_file_path} not found.")
    exit(1)

# Read the dialogues from the text file
with open(train_file_path, 'r', encoding='utf-8') as f:
    train_data = f.readlines()

def preprocess_dialogues(data):
    dialogues = []
    for dialogue in data:
        dialogues.append(dialogue.strip().split('__eou__')[:-1])
    return dialogues

train_dialogues = preprocess_dialogues(train_data)

# Custom Dataset class
class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, block_size=128):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        self.create_examples()

    def create_examples(self):
        for dialogue in self.dialogues:
            encoded = self.tokenizer.encode(' '.join(dialogue), add_special_tokens=True)
            for i in range(0, len(encoded), self.block_size):
                self.examples.append(encoded[i:i + self.block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the pad_token_id to eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Prepare the dataset for training
train_dataset = DialogueDataset(train_dialogues, tokenizer)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Function to update the model with new dialogue
def update_model_with_new_dialogue(model, tokenizer, new_dialogue, optimizer):
    model.train()
    new_input_ids = tokenizer.encode(new_dialogue, return_tensors='pt')
    outputs = model(new_input_ids, labels=new_input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save the trained model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("Training complete. The model is saved.")

# Load the saved model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./trained_model")
tokenizer = GPT2Tokenizer.from_pretrained("./trained_model")

optimizer = AdamW(model.parameters(), lr=5e-5)

# Set the pad_token_id to eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Interactive loop
print("You can now interact with the model. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Generate a response from the model
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    print(f"AI: {response}")

    # Ask if the user wants to update the model with the new information
    update = input("Do you want to update the model with this new information? (yes/no): ").strip().lower()
    if update == "yes":
        update_model_with_new_dialogue(model, tokenizer, user_input, optimizer)
        model.save_pretrained("./updated_model")
        tokenizer.save_pretrained("./updated_model")
        print("Model updated and saved.")

print("Goodbye!")
