import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import wandb

def plot_alignment(directory, directory_out, summary_only=True):
    t = []
    err = []
    iters = []
    Hs = []
    stdTrans = []
    stdRot = []
    nConstraints = []
    tv = []
    rv = []
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.csv')]
    for f_i, f in enumerate(files):
        csv_file = os.path.join(directory, f)
        df = pd.read_csv(csv_file)

        H = df['H11']
        for i in range(1, 7):
            for j in range(1, 7):
                H += df[f'H{i}{j}']
        stdTrans += [np.sqrt(df['H11'].iloc[-1]) + np.sqrt(df['H22'].iloc[-1]) + np.sqrt(df['H33'].iloc[-1])]
        stdRot += [(np.sqrt(df['H55'].iloc[-1]) + np.sqrt(df['H44'].iloc[-1]) + np.sqrt(df['H66'].iloc[-1]))/np.pi * 180.0]

        t += [float(f.replace('Alignment_', '').replace('.csv', ''))]
        err += [df['Squared Error'][len(df['Squared Error'])-1]]
        Hs += [H[len(H)-1]]
        iters += [df['Iteration'][len(df['Iteration'])-1]]
        nConstraints += [df['nConstraints'][len(df['nConstraints'])-1]]
        tv += [np.sqrt(np.square(df['tx']) + np.square(df['ty']) + np.square(df['tz'])).iloc[-1]]
        rv += [np.sqrt(np.square(df['rx']) + np.square(df['ry']) + np.square(df['rz'])).iloc[-1]/np.pi * 180.0]
        if summary_only:
            continue

        print(f"{f_i}/{len(files)}")
        plt.figure(figsize=(6, 10))
        plt.subplot(5, 1, 1)
        plt.plot(np.arange(len(df['Squared Error'])), np.array(df['Squared Error']/np.array(df['nConstraints'])))
        plt.grid('both')
        plt.title('Squared Error')
        plt.xlabel('$\\bar{Chi^2}$')
        for i, key in enumerate(['Level', 'Step Size', 'nConstraints']):
            plt.subplot(5, 1, i+2)
            plt.plot(np.arange(len(df[key])), np.array(df[key]))
            plt.grid('both')
            plt.title(key)
            plt.xlabel('Iteration')
        plt.subplot(5, 1, 5)
        plt.plot(np.arange(len(df['H11'])), H)
        plt.title('Hessian')
        plt.grid('both')
        plt.xlabel('Iteration')
        plt.tight_layout()

        plt.savefig(csv_file.replace('csv', 'png'))
        plt.close('all')
    t = np.array(t)
    t -= t[0]
    t /= 1e9
    err = np.array(err)
    Hs = np.array(Hs)
    stdTrans = np.array(stdTrans)
    stdRot = np.array(stdRot)

    nConstraints = np.array(nConstraints)
    iters = np.array(iters)
    plt.figure(figsize=(20, 6))
    plt.subplot(2, 3, 1)
    plt.title('Squared Error')
    plt.xlabel('$t - t_0 [s]$')
    plt.ylabel('$\\bar{Chi^2}$')
    plt.grid('both')
    plt.plot(t, err)

    plt.subplot(2, 3, 5)
    plt.title('Uncertainty Translation')
    plt.xlabel('$t - t_0 [s]$')
    plt.ylabel('$\\sigma_t [m]$')
    plt.grid('both')
    plt.plot(t, stdTrans)
    # plt.ylim(0, 1e-1)

    plt.subplot(2, 3, 6)
    plt.title('Uncertainty Rotation')
    plt.xlabel('$t - t_0 [s]$')
    plt.ylabel('$\\sigma_r [°]$')
    plt.grid('both')
    plt.plot(t, stdRot)
    
    plt.subplot(2, 3, 4)
    plt.title('Iterations')
    plt.grid('both')
    plt.xlabel('$t - t_0 [s]$')
    plt.ylabel('#')
    plt.plot(t, iters)

    plt.subplot(2, 3, 2)
    plt.title('Translation')
    plt.ylabel("$\\Delta t   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, tv, '.--')

    plt.subplot(2, 3, 3)
    plt.title('Rotation')
    plt.ylabel("$\\Delta r   [°]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, rv, '.--')

    plt.tight_layout()
    # plt.show()
    plt.savefig(directory_out + '/alignment.png')
    plt.close('all')
    
    metric_t = wandb.define_metric("Timestamp", summary=None, hidden=True)
    wandb.define_metric("Iterations", summary=None, hidden=False, step_metric=metric_t)
    wandb.define_metric("Alignment Error", summary='mean', goal='minimize', step_metric=metric_t)
    wandb.define_metric("Translation", summary=None, hidden=True, step_metric=metric_t)
    wandb.define_metric("Rotation", summary=None, hidden=True, step_metric=metric_t)

    for i in range(t.shape[0]):
        wandb.log({'Timestamp': t[i],
                   'Alignment Error': err[i],
                   'Iterations': iters[i],
                   'Translation': tv[i],
                   'Rotation': rv[i]})
    