#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "❌ .env file not found!"
    exit 1
fi

# Auto-login to HuggingFace
echo "🔐 Logging into HuggingFace..."
python -c "
from huggingface_hub import login
import os
token = os.getenv('HF_TOKEN')
if not token:
    print('❌ HF_TOKEN not found in environment')
    exit(1)
login(token=token, add_to_git_credential=True)
print('✅ Successfully logged into HuggingFace')
"

# Prompt for upload type
echo ""
echo "📦 What would you like to upload?"
echo "1) Adapter only (LoRA weights)"
echo "2) Full merged model"
read -p "Enter choice (1 or 2): " upload_type

# Use HF_USERNAME from .env and prompt for model name
echo ""
read -p "🏷️  Enter model name (will be uploaded as $HF_USERNAME/model-name): " model_name
full_model_name="$HF_USERNAME/$model_name"

# Set output directory
output_dir="./output/qwen3b-automotive"

if [ "$upload_type" = "1" ]; then
    echo ""
    echo "📤 Uploading adapter only..."
    python -c "
from peft import PeftModel
from transformers import AutoTokenizer
import os

model_name = '$full_model_name'
output_dir = '$output_dir'

print(f'Loading adapter from {output_dir}...')
tokenizer = AutoTokenizer.from_pretrained(output_dir)

print(f'Pushing adapter to {model_name}...')
tokenizer.push_to_hub(model_name)

# Push adapter files
import shutil
from huggingface_hub import HfApi
api = HfApi()

adapter_files = ['adapter_config.json', 'adapter_model.safetensors']
for file in adapter_files:
    file_path = os.path.join(output_dir, file)
    if os.path.exists(file_path):
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file,
            repo_id=model_name
        )

print('✅ Adapter uploaded successfully!')
"

elif [ "$upload_type" = "2" ]; then
    echo ""
    echo "📤 Uploading full merged model..."
    python -c "
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = '$full_model_name'
output_dir = '$output_dir'
base_model = 'Qwen/Qwen2.5-3B-Instruct'

print('Loading and merging model...')
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

merged_model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(output_dir)

print(f'Pushing merged model to {model_name}...')
merged_model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)

print('✅ Full model uploaded successfully!')
"

else
    echo "❌ Invalid choice. Please run again and select 1 or 2."
    exit 1
fi

echo ""
echo "🎉 Upload complete! Model available at: https://huggingface.co/$full_model_name"