import re
import matplotlib.pyplot as plt
import numpy as np

# File path to the recorded results
log_file_path = "ans_detail.log"

# Ratios used in the original script
ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Data storage for "seen" and "unseen"
data_seen = {ratio: [] for ratio in ratios}
data_unseen = {ratio: [] for ratio in ratios}

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
            if subfolder == "seen":
                data_seen[ratio].append(log_prob)
            elif subfolder == "unseen":
                data_unseen[ratio].append(log_prob)

# Calculate the average log probabilities for each ratio
avg_log_probs_seen = [np.mean(data_seen[ratio]) for ratio in ratios]
avg_log_probs_unseen = [np.mean(data_unseen[ratio]) for ratio in ratios]

# Visualization
plt.figure(figsize=(12, 8))

# Plot average log probabilities for "seen" and "unseen"
plt.plot(avg_log_probs_seen, ratios, marker='o', color='blue', label='Seen')
plt.plot(avg_log_probs_unseen, ratios, marker='o', color='red', label='Unseen')

# Customize the plot
plt.title("Log Probabilities vs. Ratios for Seen and Unseen Data")
plt.xlabel("Average Log Probability")
plt.ylabel("Ratios")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot as an image
output_plot_path = "log_probabilities_plot.png"  # Change file format if needed, e.g., .pdf
plt.savefig(output_plot_path)
print(f"Plot saved to {output_plot_path}")
