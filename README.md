# q1k_eeget_init
Initialization code for the EEG/ET data from the Q1K project.

## clone stuff
```bash
git clone https://github.com/lina-usc/q1k_eeget_init.git
```

## do the installs, etc...
```bash
cd q1k_eeget_init
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install mne
pip install -r ./q1k_eeget_init/requirements.txt
pip install -e .
```

## download the raw session files from the remote cluster...
On a local computer navigate to the project path and download the raw session files to the sourcefiles folder

For example if the "q1k" folder is in your home directory:

```bash
cd ~/q1k/pilot/q1k-external-pilot
scp -r <username>@narval.computecanada.ca:/project/def-emayada/q1k/pilot/q1k-external-pilot/sourcefiles/012_1 sourcefiles
```

## perform the init procedure...
open the q1k_eeget_init.ipynb in VSCode and follow the instructions..

## upload the BIDS raw data file to the remote cluster for pylossless processing...
If it is the first task of the session being processed upload the entire folder... e.g.
```bash
scp -r ~/q1k/pilot/q1k-external-pilot/sub-012 <username>@narval.computecanada.ca:/project/def-emayada/q1k/pilot/q1k-external-pilot
```
 If the session already has a folder on the remote cluster just upload the specific files... e.g.
```bash
scp -r ~/q1k/pilot/q1k-external-pilot/sub-012/ses-01/eeg/*_task-vp_* <username>@narval.computecanada.ca:/project/def-emayada/q1k/pilot/q1k-external-pilot/sub-012/ses-01/eeg
scp -r ~/q1k/pilot/q1k-external-pilot/sub-012/ses-01/*_scans.tsv <username>@narval.computecanada.ca:/project/def-emayada/q1k/pilot/q1k-external-pilot/sub-012/ses-01
```
... then running the pylossless pipeline on the raw BIDS session is handled by the q1k_pylossless package at... 

