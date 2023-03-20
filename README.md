# q1k_eeget_init
Initialization code for the EEG/ET data from the Q1K project.

# setup

Assumes that https://github.com/jadesjardins/mne-python.git is a fork of git@github.com:scott-huberty/mne-python.git

git clone https://github.com/lina-usc/q1k_eeget_init.git
cd q1k_eeget_init
git clone https://github.com/jadesjardins/mne-python.git
cd mne-python
git remote add scotts_fork git@github.com:scott-huberty/mne-python.git
git fetch scotts_fork
git checkout --track scotts_fork/mne-eyetrack_revisions
cd ..
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install -r mne-python/requirements.txt
pip install -e mne-python
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name q1k_init_kernel

