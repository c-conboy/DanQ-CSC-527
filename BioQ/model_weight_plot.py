import matplotlib.pyplot as plt
import numpy as np

# Layer names and their corresponding parameter counts
layers = [
    "Convolutional Layer",
    "1st LSTM Layer (Bidirectional)",
    "2nd LSTM Layer (Bidirectional)",
    "1st Fully Connected Layer",
    "2nd Fully Connected Layer",
]

parameters = [
    33280,  # Conv layer
    1632000,  # 1st LSTM
    2457600,  # 2nd LSTM
    44400000,  # 1st Dense
    850094,  # 2nd Dense
]

total_parameters = sum(parameters)

# Normalize parameter counts for percentage breakdown
percentages = [(p / total_parameters) * 100 for p in parameters]

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_positions = np.arange(len(layers))
bar_colors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#f44336']

ax.barh(bar_positions, parameters, color=bar_colors, alpha=0.8)

# Add labels and title
ax.set_yticks(bar_positions)
ax.set_yticklabels(layers)
ax.set_xlabel("Number of Parameters")
ax.set_title("DanQ Model Parameter Breakdown")

# Annotate bar chart with parameter counts
for i, value in enumerate(parameters):
    ax.text(value + total_parameters * 0.01, i, f"{value:,} ({percentages[i]:.1f}%)", va='center')

plt.tight_layout()
plt.show()
plt.savefig("model_weights.png")
