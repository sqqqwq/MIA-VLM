from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
# Define model repo on Hugging Face and cache directory

image_path = "llava_v1_5_radar.jpg"

# Load processor and model, using FP16 precision for memory efficiency
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:1")

# Load local image
image = Image.open(image_path).convert("RGB")

# Prepare conversation with the image and prompt
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Process the image and prompt inputs
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:1")
print(inputs['pixel_values'].shape)  # Shape of the tensor
#print(inputs['pixel_values'])        # Tensor values

# Generate output with scores
output = model.generate(
    **inputs,
    max_new_tokens=500,
    return_dict_in_generate=True,
    output_scores=True,
    repetition_penalty=1.2
)

# Decode the generated tokens to text
decoded_output = processor.decode(output.sequences[0], skip_special_tokens=True)
print("Generated Output:", decoded_output)

# Extract the output scores (logits) for each generation step
scores = output.scores  # List of tensors with shape (batch_size, vocab_size)

# Get the EOS token ID from the tokenizer
eos_token_id = processor.tokenizer.eos_token_id

# Initialize a list to store EOS probabilities at each time step
eos_probs = []

for score in scores:
    # Apply softmax to convert logits to probabilities
    probs = torch.nn.functional.softmax(score, dim=-1)  # Shape: (batch_size, vocab_size)
    # Extract the probability of the EOS token
    eos_prob = probs[:, eos_token_id]  # Shape: (batch_size,)
    eos_probs.append(eos_prob)

# Convert the list of tensors to a single tensor of shape (batch_size, sequence_length)
eos_probs_tensor = torch.stack(eos_probs, dim=1)

# Compute the average EOS probability for each sequence in the batch
avg_eos_prob = eos_probs_tensor.mean(dim=1)  # Shape: (batch_size,)

# Print the average EOS probability
print(f"Average EOS Probability: {avg_eos_prob.item():.4f}")

# Optional: If you want to see the EOS probability at each time step
print("EOS Probabilities at each time step:")
'''
for idx, prob in enumerate(eos_probs_tensor[0]):
    print(f"Step {idx+1}: {prob.item():.4f}")
'''
