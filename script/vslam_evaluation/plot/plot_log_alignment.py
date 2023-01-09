import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_alignment(directory, summary_only=False):
    t = []
    err = []
    iters = []
    Hs = []
    nConstraints = []
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.csv')]
    for f_i, f in enumerate(files):
        csv_file = os.path.join(directory, f)
        df = pd.read_csv(csv_file)

        H = df['H11']
        for i in range(1, 6):
            for j in range(1, 6):
                H += np.square(df[f'H{i}{j}'])
        t += [float(f.replace('Alignment_', '').replace('.csv', ''))]
        err += [df['Squared Error'][len(df['Squared Error'])-1]]
        Hs += [H[len(H)-1]]
        iters += [df['Iteration'][len(df['Iteration'])-1]]
        nConstraints += [df['nConstraints'][len(df['nConstraints'])-1]]

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
    print(H)
    t = np.array(t)
    t -= t[0]
    t /= 1e9
    err = np.array(err)
    Hs = np.array(Hs)
    nConstraints = np.array(nConstraints)
    iters = np.array(iters)
    plt.figure(figsize=(20, 6))
    plt.subplot(2, 2, 1)
    plt.title('Squared Error')
    plt.xlabel('$t - t_0 [s]$')
    plt.ylabel('$\\bar{Chi^2}$')
    plt.grid('both')
    plt.plot(t, err)
  
    plt.subplot(2, 2, 2)
    plt.title('Hs')
    plt.xlabel('$t - t_0 [s]$')
    plt.ylabel('$|H|$')
    plt.grid('both')
    plt.plot(t, Hs)
    #plt.ylim(0, 1e-1)

    plt.subplot(2, 2, 3)
    plt.title('nConstraints')
    plt.grid('both')
    plt.xlabel('$t - t_0 [s]$')
    plt.ylabel('#')
    plt.plot(t, nConstraints)
  
    plt.subplot(2, 2, 4)
    plt.title('Iterations')
    plt.grid('both')
    plt.xlabel('$t - t_0 [s]$')
    plt.ylabel('#')
    plt.plot(t, iters)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '/alignment.png')
    plt.close('all')
      