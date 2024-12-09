from torchviz import make_dot
from danq_model import DANQ
from config import device

import torch

print("Load Data")
input_tensor, _ = torch.load('./data/visualization_batch.pt', weights_only=True)

print(f"Load Model: using device: {device}")
model = DANQ().to(device)

# Generate the computation graph
print("Making Prediction as Output")
output = model(input_tensor)

print("Constructing Graph")
graph = make_dot(output, params=dict(model.named_parameters()))

print("Generating Graph image")
graph.render("danq_model", format="png", cleanup=True)
