import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import wandb


def plot_residual(directory, directory_out):
    t = []
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.jpg')]
    print(f"Found [{len(files)}] images in {directory}")
    for i, f in enumerate(files):
        t += [float(f.replace('ResidualFinal_', '').replace('.jpg', ''))]
        wandb.log({'ResidualFinal': wandb.Image(os.path.join(directory, f))})
    