import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import wandb


def create_figure(directory, directory_out, file):
    csv_file = os.path.join(directory, file)
    df = pd.read_csv(csv_file)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df['rI'], df['rZ'], df['weights'], marker='x')

    ax.set_xlabel('$r_I$')
    ax.set_ylabel('$r_Z$')
    ax.set_zlabel('$w$')

    plt.savefig(f"{directory_out}/{file.replace('csv', 'png')}")
    plt.close('all')


def log_to_wandb(directory, file):
    csv_file = os.path.join(directory, file)
    df = pd.read_csv(csv_file)
    metric_t = wandb.define_metric("Weight", summary=None, hidden=True)
    wandb.define_metric("rI", summary=None, hidden=False, step_metric=metric_t)
    wandb.define_metric("rZ", summary=None, hidden=False, step_metric=metric_t)

    for r in df.iterrows():
        wandb.log({'Weight': r['weight'],
                    'rI': r['rI'],
                    'rZ': r['rZ']})


def plot_weights(directory, directory_out, summary_only=False,upload=False):
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.csv')]
    if not summary_only:
        for f_i, f in enumerate(files):
            create_figure(directory, directory, f)
    else:
        create_figure(directory, directory_out, files[-1])
        if upload:
            log_to_wandb(directory, files[-1])