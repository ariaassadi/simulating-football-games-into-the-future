# Simulating Football Games into the Future

This directory contains the code for Aria Assadi's Master Thesis at Uppsala University. The goal of the project is to create a model that can predict football games two seconds into the future.

## Initial Data Setup

The data is stored on an Alvis virtual machine, and the code is primarily designed to be executed on the same VM. To create a virtual machine, please read the instructions in `../alvis-instructions/Alvis-general-instructions.md`. When your VM is created, please do the following if the data is not stored on Alvis:

```bash
cd ../../
git clone https://github.com/twelvefootball/twelve-signality-wyscout.git
vim twelve-signality-wyscout/settings.py
```

Specify SIGNALITY_USERNAME, SIGNALITY_PASSWORD, and the path to the storage directory:

```python
ROOT_DIR = '/mimer/NOBACKUP/groups/two_second_football'
DATA_LOCAL_FOLDER = '/mimer/NOBACKUP/groups/two_second_football'
```

For security reasons, the username and password are kept by Twelve Football. Get in touch with them if you need the credentials.

Now you can run the following to create the data files needed:

```bash
python3 twelve-signality-wyscout/alvis/download_signality_data
python3 twelve-signality-wyscout/alvis/create_frames_df
```

It will take a couple of hours to run the scripts. When the scripts are completed, the data is ready.
