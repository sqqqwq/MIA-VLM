import os
from PIL import Image

# Function to blacken the right half of an image
def blacken_right_half(image_path, output_path):
    # Open the image
    img = Image.open(image_path)
    img = img.convert("RGB")  # Ensure it's in RGB mode
    
    # Get image dimensions
    width, height = img.size

    # Create a black rectangle of the same height as the image
    black_area = Image.new("RGB", (width // 2, height), (0, 0, 0))
    
    # Paste the black rectangle on the right half of the image
    img.paste(black_area, (width // 2, 0))

    # Save the modified image
    img.save(output_path)

# Function to process all images in a folder
def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        
        # Check if the file is an image
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            blacken_right_half(input_path, output_path)

# Main script
def main():
    # Define the input and output folder mappings
    folders_to_process = {
        "images/seen": "black/seen",
        "images/unseen": "black/unseen"
    }

    # Process each folder
    for input_folder, output_folder in folders_to_process.items():
        print(f"Processing {input_folder}...")
        process_folder(input_folder, output_folder)
    print("Processing complete. All images have been saved with the right half blackened.")

if __name__ == "__main__":
    main()
