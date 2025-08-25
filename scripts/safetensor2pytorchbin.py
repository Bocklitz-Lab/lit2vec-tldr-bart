from transformers import AutoModelForSeq2SeqLM

# Load from the safetensors-based directory
model = AutoModelForSeq2SeqLM.from_pretrained("saved_data/final_model", trust_remote_code=True)

# Re-save with standard PyTorch format
model.save_pretrained("saved_data/final_model", safe_serialization=False)
