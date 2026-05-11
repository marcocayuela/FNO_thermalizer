# FNO_thermalizer
FNO + thermalizer

`src` contains all the necessary code to perform the experiment.

The folder `fno` contains the FNO-like architectures used in this experiment:
  - `fno_2D.py` implements the standard FNO architecture.
  - `fno_2D_classifier.py` implements the standard FNO architecture augmented with an additional classifier output.

The folder `training` contains all the tools and processes required to train the different models:
  - Emulator:
    - `fno_training.py`
    - `trainer.py`

  - Diffusion model:
    - `DiffusionModel.py`
    - `diffusion_training.py`
    - `trainer_diffusion.py`

  - Common utilities:
    - `factory.py`
    - `dataset_manager.py`
    - `metric_logger.py`

The data must be stored in a folder named `Data` located at the root of the project.

The simulations can be generated using the KolSol solver available at:
https://github.com/MagriLab/KolSol

The generated simulations should be saved as `.h5` files in:

`Data/kolmogorov/Re90/train_traj/`

to match the structure expected by `dataset_manager.py`.

To run the experiment:
  - train the emulator by running `main_emul.py`
  - train the diffusion model by running `main_diff.py`

All model hyperparameters can be configured in:
  - `config_command_emul.yaml`
  - `config_command_diff.yaml`
