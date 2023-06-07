import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# from scipy.spatial.transform import Rotation as R


def plot_kalman(directory, directory_out, summary_only=True,upload=False):
    f = [f for f in sorted(os.listdir(directory)) if f.endswith('.csv')][-1]
    print(f"Reading: {f}")
    df = pd.read_csv(os.path.join(directory, f))
    #df = df.tail(-1)
    t = df['Timestamp']/1e9
    print(t)
    #print(np.array(df['Timestamp'],dtype=float)/1e9)
    # df['R'] = df.apply(lambda x: R.from_rotvec(df['rx'],df['ry'],df['rz']))
    # df['vR'] = df.apply(lambda x: R.from_rotvec(df['vrx'],df['vry'],df['vrz'])) 
    df['tv'] = np.sqrt(np.square(df['vtx']) + np.square(df['vty']) + np.square(df['vtz']))
    df['tvc'] = np.sqrt((df['cov66']+df['cov77']+df['cov88']))
    # df['tve'] = df.apply(lambda x: np.sqrt((df['cov77']+df['cov88'],df['cov99'])*2))
    df['rv'] = np.sqrt(np.square(df['vrx']) + np.square(df['vry']) + np.square(df['vrz']))
    df['rvc'] = np.sqrt((df['cov99']+df['cov1010']+df['cov1111']))
    # df['rve'] = df.apply(lambda x: np.sqrt((df['cov1010']+df['cov1111'],df['cov1212'])*2))
    df['et'] = np.sqrt(np.square(df['evtx']) + np.square(df['evty']) + np.square(df['evtz']))
    df['etc'] = np.sqrt((df['ecov00']+df['ecov11']+df['ecov22']))
    df['mt'] = np.sqrt(np.square(df['mvtx']) + np.square(df['mvty']) + np.square(df['mvtz']))
    df['mtc'] = np.sqrt((df['mcov00']+df['mcov11']+df['mcov22']))
    df['er'] = np.sqrt(np.square(df['evrx']) + np.square(df['evry']) + np.square(df['evrz']))
    df['erc'] = np.sqrt((df['ecov33']+df['ecov44']+df['ecov55']))
    df['mr'] = np.sqrt(np.square(df['mvrx']) + np.square(df['mvry']) + np.square(df['mvrz']))
    df['mrc'] = np.sqrt((df['mcov33']+df['mcov44']+df['mcov44']))
    t = np.array(t)
    t -= t[0]
    #t /= 1e9
    plt.figure(figsize=(10, 10))
    plt.ylabel("$t_y   [m]$")
    plt.xlabel("$t_x   [m]$")
    plt.grid('both')
    plt.plot(df['px'], df['py'], '.--')
    plt.axis('equal')
    plt.savefig(directory_out + '/trajectory_xy_kalman.png')
    
    plt.figure(figsize=(10, 10))
    plt.ylabel("$t_z   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['pz'])
    plt.savefig(directory_out + '/trajectory_z_kalman.png')

    plt.figure(figsize=(20, 6))

    plt.subplot(2, 4, 1)
    plt.ylabel("$v_t   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['tv'], '.--')
    # plt.errorbar(t, y = df['tv'],yerr=df['tve'])
    plt.subplot(2, 4, 2)
    plt.ylabel("$\\Sigma_t   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['tvc'], '.--')
   
    plt.subplot(2, 4, 3)
    plt.ylabel("$\\Delta t   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['et'], '.--')
    plt.plot(t, df['mt'], '.--')
    plt.legend(["Expectation", "Measurement"])
    
    plt.subplot(2, 4, 4)
    plt.ylabel("$\\Sigma_t   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['etc'], '.--')
    plt.plot(t, df['mtc'], '.--')
    plt.legend(["Expectation", "Measurement"])
    
    plt.subplot(2, 4, 5)
    plt.ylabel("$v_r   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['rv'], '.--')
    # plt.errorbar(t, y = df['rv'],yerr=df['rve'])
    plt.subplot(2, 4, 6)
    plt.ylabel("$\\Sigma_r   [°]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['rvc'], '.--')
  
    plt.subplot(2, 4, 7)
    plt.ylabel("$\\Delta r   [°]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['er']/np.pi*180.0, '.--')
    plt.plot(t, df['mr']/np.pi*180.0, '.--')
    plt.legend(["Expectation", "Measurement"])
    
    plt.subplot(2, 4, 8)
    plt.ylabel("$\\Sigma_r   [°]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid('both')
    plt.plot(t, df['erc'], '.--')
    plt.plot(t, df['mrc'], '.--')
    plt.legend(["Expectation", "Measurement"])
    
        
    # plt.show()
    plt.savefig(directory_out + '/kalman.png')
    plt.close('all')
      