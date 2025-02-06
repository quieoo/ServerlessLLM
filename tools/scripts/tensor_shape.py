import torch
from transformers import AutoModel, AutoTokenizer

# Load the model
model_path = "/mnt/n0/models/opt1.3"
model = AutoModel.from_pretrained(model_path)

# Load the tokenizer (if needed)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Accessing the model's state_dict (tensors)
model_state_dict = model.state_dict()

# Print all tensor keys in the model
for key, tensor in model_state_dict.items():
    # get tensor memory consumption in bytes
    tensor_memory = tensor.element_size() * tensor.nelement()
    print(f"Tensor name: {key}, Shape: {tensor.shape}, Memory: {tensor_memory/1024:.2f} KB")
