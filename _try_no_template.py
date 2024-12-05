from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

# Load processor and model
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

# Set patch_size and vision_feature_select_strategy as attributes (do not pass them as kwargs)
processor.patch_size = 4  # Adjust patch size based on model's requirements
processor.vision_feature_select_strategy = "default"  # Ensure this matches the model's config

# Load the image
image = Image.open("llava_v1_5_radar.jpg")

# Prepare the image and add a textual input (prompt)
prompt = "What is shown in this image?"
inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt"
)

# Print out each input_id and its corresponding token (without special tokens)
for idx, token_id in enumerate(inputs['input_ids'][0]):
    token = processor.decode([token_id], skip_special_tokens=True)  # Skip special tokens for clarity
    print(f"ID: {token_id.item()} Token: {token}")

# Print number of images
print("Number of Images:", len(inputs['pixel_values']))

# Generate caption with appropriate generation parameters
output = model.generate(
    **inputs,
    max_new_tokens=50,  # Set a limit for caption length
    temperature=0.7,     # Control randomness
    do_sample=True       # Enable sampling for more diverse captions
)

# Decode and print the generated caption
caption = processor.decode(output[0], skip_special_tokens=True)
print("Generated Caption:", caption)
