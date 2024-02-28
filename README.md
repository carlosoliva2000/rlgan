# Credits

- Carlos Oliva López
- Christian Graf Aray


# Installation

These scripts are intendeed to be used, were tested and executed with Python 3.9.18 on Windows 10. This is due to the tensorflow gpu limitation on newer Python versions on Windows. We can't guarantee these scripts will work using other Python versions. If you have any subversion of Python 3.9 already installed, you may want to try it first before installing 3.9.18.

To install the dependencies, we recommend the use of **conda** ([miniconda](https://docs.anaconda.com/free/miniconda/index.html) is a lightweight option). You can create a conda environment with Python 3.9.18 typing:

`conda create --name <desired_env_name> --file conda_requirements.txt`.

If you don't have conda installed or don't want to use it (although is highly recommended), you can try creating a virtual environment (or in your default Python installation if you have the version 3.9.18) typing:

1. **Optional:**  only if you want to create a virtual environment (change the path to where Python 3.9.18 is installed). `path/to/python3.9.18/python3 -m venv <desired_env_name>`
2. **Optional:** only if you followed the previous step. `source <env_name>/Scripts/activate` in Windows. `source <env_name>/bin/activate` in UNIX.
3. `pip install -r requirements.txt`

These scripts were tested and used in Windows 10. They have not been tested in any UNIX OS (Linux, Ubuntu, etc), therefore some dependencies might (or not) be problematic.

# Execution

Remember to activate your conda `conda activate <env_name>` or virtual environment (step 2 in the previous section) before executing any script.

Each subproject (P1, P2 and P3) have a file `_EXECUTE_FROM_THIS_PATH.txt` for a reason. For example, if you want to train/test a model from P2, set your current working directory to that directory. Then, you can simply call `./test.py` or `./train.py`.


# Disclaimer

The versions of Neuroevolution (P1) and DQN (P2/DQN) have been developed by Carlos.

DuelingDQN (P2/DuelingDQN) has been developed by Christian and the instructions above don't really apply, because it uses PyTorch instead of TensorFlow.

- To train: main.py
- To create charts: main_load_charts.py
- To test: main_test_networks.py
