# Instructions for How to Run ALVIS

This guide aims to explain how to set up ALVIS in order to run the repositories provided for the football-related master theses at Uppsala University 2024. In order to following this guide, you need to be connected to eduroam.

## Join the Alvis Project

Send you university email to David Sumpter and he will add you to the project.

Log in to SUPR and check that you have been added to the project:
[https://supr.naiss.se/](https://supr.naiss.se/)

## Set up Virtual Environment and Bash Script

Open your Alvis Dashboard:
[https://portal.c3se.chalmers.se/pun/sys/dashboard/batch_connect/sys/bc_desktop/session_contexts/new](https://portal.c3se.chalmers.se/pun/sys/dashboard/batch_connect/sys/bc_desktop/session_contexts/new)

Open your terminal and do the following to create a virtual environment:
```bash
mkdir ~/master-thesis
cd ~/master-thesis
module load SciPy-bundle/2021.10-foss-2022a matplotlib/3.5.2-foss-2022a JupyterLab/3.5.0-GCCcore-11.3.0
virtualenv --system-site-packages my_python
source my_python/bin/activate
pip install --no-cache-dir --no-build-isolation mplsoccer
pip install --no-cache-dir --no-build-isolation pyarrow==15.0.0
python -m ipykernel install --user --name=my_python --display-name="My Python"
```
Now we need to create a bash script that will tell Jupyter how to access our modules:
```bash
mkdir ~/portal/jupyter/
vim ~/portal/jupyter/virtual-setup.sh   # TODO: Check if this name is correct
```
Paste the following content to the bash file:
```bash
# TODO: Paste everything here
```

## Configure Jupyter
From the Alvis Dashboard, start a Jupyter session. Ensure the session folder starts from the directory "/master-thesis". Select our newly created bash script "/portal/jupyter/virtual-setup.sh" when configuring.

The kernel can be selected from the top right corner, and you should select "My Python".