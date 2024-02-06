# Instructions for How to Run ALVIS

This guide aims to explain how to set up ALVIS in order to run the repositories provided for the football-related master theses at Uppsala University 2024. Hopefully this guide will be enough, but you can find more information at [https://www.c3se.chalmers.se/documentation/modules/](https://www.c3se.chalmers.se/documentation/modules/).

## Join the Alvis Project

Send you university email to David Sumpter and he will add you to the project.

Log in to SUPR and check that you have been added to the project by going to [https://supr.naiss.se/](https://supr.naiss.se/).

If you are using SUPR for the first time, go to Accounts and click on `Request account at ALVIS C3SE`. It may take one working day for your account to be approved. When the account is approved, everything will be ready to use after a few hours.

## Set up Virtual Environment and Bash Script

(From now on, you need to be connected to eudoraom)

Open your Alvis Dashboard:
[https://portal.c3se.chalmers.se/pun/sys/dashboard/batch_connect/sys/bc_desktop/session_contexts/new](https://portal.c3se.chalmers.se/pun/sys/dashboard/batch_connect/sys/bc_desktop/session_contexts/new)

Open your terminal from `Clusters` -> `>_Alvis Shell Acces`. Now do the following to create a virtual environment:
```bash
mkdir ~/master-thesis
cd ~/master-thesis
module load SciPy-bundle/2021.10-foss-2022a matplotlib/3.5.2-foss-2022a JupyterLab/3.5.0-GCCcore-11.3.0
virtualenv --system-site-packages my_python
source my_python/bin/activate
pip install --no-cache-dir --no-build-isolation mplsoccer
pip install --no-cache-dir --no-build-isolation pyarrow==15.0.0
python -m ipykernel install --user --name=my_python --display-name="My Python"
````
Now we need to create a bash script that will tell Jupyter how to access our modules:
```bash
mkdir -p ~/portal/jupyter/
cd ~/portal/jupyter/
vim alvis-jupyter.sh
```
Paste the following content to the bash file:
```bash
# Ensure we don't have any conflicting modules loaded
ml purge
# Load the following modules. This will install many packages
module load SciPy-bundle/2021.10-foss-2022a matplotlib/3.5.2-foss-2022a JupyterLab/3.5.0-GCCcore-11.3.0
# You can launch jupyter, but you must specify the config file as below: 
jupyter lab --config="${CONFIG_FILE}"
```
## Start the Jupyter Session

(Next time you only need to do this step)

From the Alvis Dashboard, start the Jupyter session from `My Interactive Sessions` -> `Jupyter`.
* Under Resource, select `V1oo:1` (or any other resource of your choice)
* Under Runtime, select your newly created bash script `~/portal/vscode/alvis-jupyter.sh`
* Under Working Directory, select your `/master-thesis` folder

Start the session and connect to Jupyter. In the Jupyter Launcher, select "My Python".

Now everything is configured! Happy coding
