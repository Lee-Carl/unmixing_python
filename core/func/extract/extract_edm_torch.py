import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract_edm_torch(y, a):
    """
    Args:
        y (torch.tensor): Mixed pixels (L, N).
        a (torch.tensor): Estimated abundances (P, N).

    Returns:
        E_solution (torch.tensor): Estimated endmembers (L, P).
    """
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move data to GPU
    Y = y.float().clone().detach().to(device)
    A = a.float().clone().detach().to(device)

    # Initialize endmembers using Xavier initialization and move parameters to GPU
    E = nn.Parameter(torch.empty(Y.shape[0], A.shape[0]).to(device))
    nn.init.xavier_uniform_(E)

    # Define optimizer
    optimizer = torch.optim.Adam([E], lr=0.01)

    # Perform optimization
    for epoch in range(1000):
        optimizer.zero_grad()  # Clear gradients
        # Calculate the mean squared error loss as the objective function
        loss = F.mse_loss(Y, torch.matmul(E, A))
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        E.data = torch.clamp(E.data, min=0)  # Force E to be non-negative

    # Get the final estimated endmembers
    E_solution = E.data.clone().to(device)  # Move the result back to CPU

    return E_solution