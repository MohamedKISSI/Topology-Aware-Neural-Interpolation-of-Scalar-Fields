# Environment Setup Instructions

Before running the Python scripts, make sure to set up your environment properly. This includes installing required packages and dependencies such as **TTK** and **ParaView**.

## (Optional) Create a Virtual Environment

It is recommended to use a virtual environment to avoid conflicts with other Python projects.

```bash
python3 -m venv myenv
source myenv/bin/activate
```

## Create Result Folders

These folders will be used to store the output of the notebook:

```bash 
mkdir result_folder
mkdir result_folder_interpolation
mkdir intermediate_file
```

## Update System and Install TTK & ParaView

If you do **not** have ParaView with TTK installed on your system, you will need to install them manually.

You can download and install pre-built packages for Ubuntu 22.04:

```bash 
wget https://github.com/topology-tool-kit/ttk/releases/download/1.3.0/ttk-1.3.0-ubuntu-22.04.deb
wget https://github.com/topology-tool-kit/ttk-paraview/releases/download/v5.13.0/ttk-paraview-v5.13.0-ubuntu-22.04.deb

sudo apt update

chmod 644 ./ttk-paraview-v5.13.0-ubuntu-22.04.deb
sudo apt install -y ./ttk-paraview-v5.13.0-ubuntu-22.04.deb

chmod 644 ./ttk-1.3.0-ubuntu-22.04.deb
sudo apt install -y ./ttk-1.3.0-ubuntu-22.04.deb

```
Skip this step if ParaView with TTK plugins is already installed on your machine.

## Install Python Dependencies

```bash 
pip install numpy matplotlib torch
pip install pyvista --no-deps
```

Once everything is installed, you can launch the Jupyter Notebook. 

## Notes

These instructions assume you are using Ubuntu 22.04.

# Model Usage 

A pre-trained model for the vonKarman dataset is already provided in the **result_folder**.

To test it directly, simply run:

```bash 
python3 test.py
```

If you prefer to train the model yourself, you can do so with the following command:

```bash 
python3 train.py
```

Then, to test the model you just trained, run:

```bash
python3 test.py
```

The **test.py** script performs the following actions:

- Generates model predictions and stores them in the **result_folder**.
- Saves the trained model in the same **result_folder**.
- Generates baseline predictions using interpolation (without keyframes) and stores them in **result_folder_interpolation**.
- Produces a comparison table between the two approaches (Interpolation vs. Our model) using L2, PSNR, and W2 metrics.

This comparison table is saved as a CSV file named resultats.csv in the code directory.