# q1k_eeget_init
Initialization code for the EEG/ET data from the Q1K project.

## clone stuff
```bash
git clone https://github.com/lina-usc/q1k_eeget_init.git

git clone https://github.com/mne-tools/mne-python
```

## do the installs, etc...
```bash
cd ..
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install -e mne-python
pip install -r ./q1k_eeget_init/requirements.txt
pip install -e .
```
