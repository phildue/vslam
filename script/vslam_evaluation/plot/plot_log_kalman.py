import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# from scipy.spatial.transform import Rotation as R


def plot_kalman(directory, summary_only=True):
    f = [f for f in sorted(os.listdir(directory)) if f.endswith('.csv')][-1]
    df = pd.read_csv(os.path.join(directory, f))
    t = df['Timestamp']
    # df['R'] = df.apply(lambda x: R.from_rotvec(df['rx'],df['ry'],df['rz']))
    # df['vR'] = df.apply(lambda x: R.from_rotvec(df['vrx'],df['vry'],df['vrz'])) 
    df['tv'] = np.sqrt(np.square(df['vtx']) + np.square(df['vty']) + np.square(df['vtz']))
    # df['tve'] = df.apply(lambda x: np.sqrt((df['cov77']+df['cov88'],df['cov99'])*2))
    df['rv'] = np.sqrt(np.square(df['vrx']) + np.square(df['vry']) + np.square(df['vrz']))
    # df['rve'] = df.apply(lambda x: np.sqrt((df['cov1010']+df['cov1111'],df['cov1212'])*2))
    df['et'] = np.sqrt(np.square(df['evtx']) + np.square(df['evty']) + np.square(df['evtz']))
    df['mt'] = np.sqrt(np.square(df['mvtx']) + np.square(df['mvty']) + np.square(df['mvtz']))
    df['er'] = np.sqrt(np.square(df['evrx']) + np.square(df['evry']) + np.square(df['evrz']))
    df['mr'] = np.sqrt(np.square(df['mvrx']) + np.square(df['mvry']) + np.square(df['mvrz']))
    t = np.array(t, dtype=float)
    t -= t[0]
    t /= 1e9
    plt.figure(figsize=(20, 6))

    plt.subplot(2, 3, 1)
    plt.ylabel("$t_y   [m]$")
    plt.xlabel("$t_x   [m]$")
    plt.grid('both')
    plt.plot(df['px'], df['py'], '.--')

    plt.subplot(2, 3, 2)
    plt.ylabel("$t_z   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['pz'])

    plt.subplot(2, 3, 3)
    plt.ylabel("$v_t   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['tv'], '.--')
    # plt.errorbar(t, y = df['tv'],yerr=df['tve'])

    plt.subplot(2, 3, 4)
    plt.ylabel("$v_r   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['rv'], '.--')
    # plt.errorbar(t, y = df['rv'],yerr=df['rve'])

    plt.subplot(2, 3, 5)
    plt.ylabel("$\\Delta t   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['et'], '.--')
    plt.plot(t, df['mt'], '.--')
    plt.legend(["Expectation", "Measurement"])
    plt.subplot(2, 3, 6)
    plt.ylabel("$\\Delta r   [°]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['er']/np.pi*180.0, '.--')
    plt.plot(t, df['mr']/np.pi*180.0, '.--')
    plt.legend(["Expectation", "Measurement"])
    
    # plt.show()
    plt.savefig(directory + '/kalman.png')
    plt.close('all')
      