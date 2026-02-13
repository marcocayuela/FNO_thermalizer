import h5py
import numpy as np
import os 


def load_simulation(data_path, time_ds=1, space_ds=1):
    '''
    Return the velocity field contained at the given file_path. It has to be a h5 file with the strcuture of the code KolSol
    Shape: [nt//time_ds, nx//space_ds, ny//space_ds,2]
    
    :param data_path: path where velocity field is contained
    :param time_ds: downsampling factor across time dimension
    :param space_ds: downsampling factore across space dimensions
    '''

    with h5py.File(data_path, "r") as f:
        velocity_field = f["velocity_field"][()][::time_ds,::space_ds, ::space_ds]
    return velocity_field


def load_multiple_simulation(data_rep, time_ds=1, space_ds=1, list=None):
    '''
    Load and return several simulations contain in the same repository. (They have to have same size)
    Shape: [n_simul, nt//time_ds, nx//space_ds, ny//space_ds,2]
    
    :param data_rep: Repository where simulations are saved.
    :param time_ds: downsampling factor across time dimension
    :param space_ds: downsampling factore across space dimensions
    :param list: (list of int) If given, load only simulations correpsonding to the given number
    '''


    if list is None:
        simulation_files = [os.path.join(data_rep, f) for f in os.listdir(data_rep) if f.endswith(".h5")]
    else:
        simulation_files = [os.path.join(data_rep, f)for f in os.listdir(data_rep) 
                            if f.startswith("sim") and f.endswith(".h5") and (int(f[3:-3]) in list)]
    
    simulation_velocities = []
    for file in simulation_files:
        with h5py.File(file, 'r') as f:
            simulation_velocities.append(f["velocity_field"][()][::time_ds,::space_ds, ::space_ds])
    
    return np.stack(simulation_velocities, axis=0)


def get_info(data_path, print=True):

    '''
    Get the info and hyperparameters used for a kolmogorov simulation
    
    :param data_path: path where the simulation is located
    :param print: if true print the info
    '''

    infos = {}
    with h5py.File(data_path, "r") as f:
        infos["re"] = f["re"][()]
        infos["resolution"] = f["resolution"][()]
        infos["dt_simul"] = f["dt"][()]
        infos['simulation_time'] = f["time"][-1] + f["dt"]
        infos["nt"] = f["velocity_field"][()].shape[0]
        infos["dt_saved"] = infos["simulation_time"]//infos["nt"]
        infos["nf"] = f["nf"][()]
        infos["nk"] = f["nk"][()]

    if print:
        for key, value in infos.items():
            print(key, ":", value)

    return infos


        