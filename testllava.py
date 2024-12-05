import logging
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import os
import numpy as np  # For sorting and numerical operations
from tqdm import tqdm  # For progress bar

# Set up logging
logging.basicConfig(filename='model_output.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Define model repo on Hugging Face and cache directory
base_image_path = "black"  # Root directory containing 'seen' and 'unseen' subdirectories
subdirectories = ["seen", "unseen"]  # Subfolders to process

# Load processor and model, using FP16 precision for memory efficiency
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:1")

# Collect all images to process and their paths
image_paths = []
for subdir in subdirectories:
    image_folder = os.path.join(base_image_path, subdir)
    for image_name in os.listdir(image_folder):
        if image_name.lower().endswith('.jpg'):  # Include only .jpg files (case-insensitive)
            image_paths.append((subdir, os.path.join(image_folder, image_name)))

# Iterate through the images with tqdm progress bar
for idx, (subdir, image_path) in enumerate(tqdm(image_paths, desc="Processing Images", unit="image")):
    image_name = os.path.basename(image_path)

    # Log the image being processed
    logging.info(f"Processing Image: {subdir}/{image_name}")

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Prepare conversation with the image and prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Please describe this image in details."},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)

    # Process the image and prompt inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:1")

    # Perform inference within a no_grad context
    with torch.no_grad():
        # Generate output with scores
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            return_dict_in_generate=True,
            output_scores=True,
            repetition_penalty=1.2
        )

    # Log the output sequence and corresponding tokens
    generated_tokens = output.sequences[0].tolist()

    # Find the first token after two occurrences of padding tokens (token ID 13)
    padding_token_id = 13
    padding_count = 0
    first_non_padding_index = None
    for i, token_id in enumerate(generated_tokens):
        if token_id == padding_token_id:
            padding_count += 1
            if padding_count >= 2:
                first_non_padding_index = i
                break

    # Handle case where no valid token is found
    if first_non_padding_index is None:
        logging.warning("No token found after two occurrences of padding tokens!")
        first_non_padding_index = len(generated_tokens)  # Default to end of list

    # Extract the response tokens (after two padding tokens)
    generated_response_tokens = generated_tokens[first_non_padding_index:]

    # Decode and calculate log probabilities for tokens
    all_log_probs = []
    for i, token_id in enumerate(generated_response_tokens):
        if i < len(output.scores):
            token_log_probs = torch.nn.functional.log_softmax(output.scores[i], dim=-1)  # Apply log_softmax to get log probabilities
            token_log_prob = token_log_probs[0, token_id].item()  # Log probability of the current token
            all_log_probs.append(token_log_prob)

    # Calculate the average log probability for the top k% minimum log probabilities
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    all_log_probs_np = np.array(all_log_probs)  # Convert to numpy array for sorting
    for ratio in ratios:
        k_length = int(len(all_log_probs_np) * ratio)
        topk_log_probs = np.sort(all_log_probs_np)[:k_length]
        avg_min_log_prob = np.mean(topk_log_probs)
        logging.info(f"{subdir}/{image_name} - Average Log Probability of Bottom {ratio*100:.0f}% Tokens: {avg_min_log_prob:.4f}")

    # Decode the generated response text
    decoded_output = processor.decode(generated_response_tokens, skip_special_tokens=False)
    logging.info(f"{subdir}/{image_name} - Generated Output: {decoded_output}")
