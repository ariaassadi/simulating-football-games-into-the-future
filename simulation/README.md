# Simulating Football Games into the Future

This directory contains the code for Aria Assadi's Master Thesis at Uppsala University. The goal of the project is to create models that can predict football games two seconds into the future.

## Virtual Machine

Due to the size of this project, we leverage compute power from Alvis, a supercomputer located in Gothenburg, Sweden. To create a virtual machine, please read the instructions in `../alvis-instructions/Alvis-general-instructions.md`. By using the virtual machine, you don't need to do any data setup, as the data is already uploaded on Alvis.

## Initial Data Setup

Preparing the data for this project requires some work. The processed files for the Allsvenskan 2023 alone, make up approximately 37 GB of storage. Furthermore, data can't be shared since the trackings from Signality are not for free. However, this chapter aims to describe how a person with the right permissions can create a directory with the desired data.

### Create a Folder

The directory where we store the data can be located anywhere outside the repository; we just need to define the absolute path. From now on, we will call this directory `<data_path>`. For the Alvis machine, the path is:

```bash
data_path = '/mimer/NOBACKUP/groups/two_second_football'
```

Please create your data directory and specify it in this folder's settings file.

### Create Unprocessed Frames

In this step, we need to fetch the data from Signality, and prepare parquet files with tracking data for each game. Thankfully, there is a GitHub repository dedicated to this. Everything we need for the data of this project is available if we clone the repository and navigate to the 'alvis/' folder:

```bash
git clone https://github.com/twelvefootball/twelve-signality-wyscout
cd alvis
```

Start by modifying the settings by adding our `<data_path>`. We also have to fill in the credentials for Signality, which can be obtained by the staff at Twelve Football.

```bash
vim ../settings.py
ROOT_DIR = <data_path>
DATA_LOCAL_FOLDER = <data_path>
SIGNALITY_USERNAME = <username>
SIGNALITY_PASSWORD = <password>
:wq
```

Start by running the script that downloads the Signality data in JSON format. Then run the script for creating the desired frames in parquet format.

```bash
python3 download_signality_data.py
python3 create_frames_df.py
```
Now we should have the following file structure in `<data_path>`:

```plaintext
data/
    2022/Allsvenskan/unprocessed/
        match_id_1.parquet
        ...
    2023/Allsvenskan/unprocessed/
        match_id_1.parquet
        ...

signality/
    2022/Allsvenskan/
        tracks/
            match_id_1.json
            ...
        events/
            match_id_1.json
            ...
        match_id_1.json
        ...
    2023/Allsvenskan/
        tracks/
            match_id_1.json
            ...
        events/
            match_id_1.json
            ...
        match_id_1.json
        ...
```

### Create Processed Frames

Now it is time to locate back to this repository, 'simulating-football-games-into-the-future'. We need to add features to our data to perform tasks such as model training, model evaluation, etc. These features can be added with the 'add-features.ipynb' notebook. Adding all the features to all games takes approximately three hours. To add all features to all games, simply run the associated script:

```bash
./run_add_features.sh
```

The data directory should look like this:
```plaintext
data/
    2022/Allsvenskan/processed/
        match_id_1.parquet
        ...
    2023/Allsvenskan/processed/
        match_id_1.parquet
        ...
    2022/Allsvenskan/unprocessed/
        match_id_1.parquet
        ...
    2023/Allsvenskan/unprocessed/
        match_id_1.parquet
        ...

signality/
    ...
```

Now your data is prepared and you should have everything you need to start utilizing the intricacies of this folder. Good luck!
