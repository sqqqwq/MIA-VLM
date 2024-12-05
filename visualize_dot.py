import re
import matplotlib.pyplot as plt
import numpy as np

# File path to the recorded results
log_file_path = "ans_detail.log"

# Choose one ratio to focus on
target_ratio = 0.1  # For example, 10%
random_y_offset_range = 0.05  # Control the range of random offsets for y-values

# Data storage for the chosen ratio
data_seen = []
data_unseen = []

# Regular expression to parse log lines
pattern = re.compile(r"(seen|unseen)/(.+?) - Average Log Probability of Bottom (\d+)% Tokens: (-?\d+\.\d+)")

# Read and parse the log file
with open(log_file_path, "r") as log_file:
    for line in log_file:
        match = pattern.search(line)
        if match:
            subfolder = match.group(1)  # "seen" or "unseen"
            ratio_percent = int(match.group(3))  # e.g., "5", "10"
            log_prob = float(match.group(4))  # Log probability value
            ratio = ratio_percent / 100  # Convert to decimal (e.g., "0.05")
            if ratio == target_ratio:  # Focus on the target ratio
                if subfolder == "seen":
                    data_seen.append(log_prob)
                elif subfolder == "unseen":
                    data_unseen.append(log_prob)

# Generate random y-values for jittering
random_y_seen = np.random.uniform(-random_y_offset_range, random_y_offset_range, len(data_seen))
random_y_unseen = np.random.uniform(-random_y_offset_range, random_y_offset_range, len(data_unseen))

# Visualization
plt.figure(figsize=(12, 8))

# Scatter plot for "seen"
plt.scatter(data_seen, random_y_seen, color='blue', label='Seen')

# Scatter plot for "unseen"
plt.scatter(data_unseen, random_y_unseen, color='red', label='Unseen')

# Customize the plot
plt.title(f"Log Probabilities for Ratio {target_ratio*100:.0f}%")
plt.xlabel("Log Probability")
plt.ylabel("Randomized Y-Offset")
plt.axvline(np.mean(data_seen), color='blue', linestyle='--', label='Seen Mean')
plt.axvline(np.mean(data_unseen), color='red', linestyle='--', label='Unseen Mean')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot as an image
output_plot_path = f"log_probabilities_ratio_{int(target_ratio*100)}.png"
plt.savefig(output_plot_path)
print(f"Scatter plot saved to {output_plot_path}")
