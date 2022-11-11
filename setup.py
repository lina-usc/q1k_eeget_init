# Authors: 
# James Desjardins <jim.a.desjardins@gmail.com>
# Scott Huberty <scott.huberty@mail.mcgill.ca>
# Diksha Srishyla: <diksha.srishyla@mail.mcgill.ca>
# Christian O'Reilly <christian.oreilly@gmail.com>;

# License: MIT

from setuptools import setup


if __name__ == "__main__":
    hard_dependencies = ('numpy', 'scipy', 'mne', 'mne-bids')
    install_requires = list()
    with open('requirements.txt', 'r') as fid:
        for line in fid:
            req = line.strip()
            for hard_dep in hard_dependencies:
                if req.startswith(hard_dep):
                    install_requires.append(req)

    setup(
        name='q1kinit',
        version="0.0.1",
        description='Initialization code for the EEG/ET data from the Q1K project.',
        python_requires='>=3.5',
        author="James Desjardins",
        author_email='jim.a.desjardins@gmail.com',
        url='https://github.com/lina-usc/q1k_eeget_init',
        packages=['q1kinit'],
        install_requires=install_requires)
