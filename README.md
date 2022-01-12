# Pong game using Duel Deep Q-learning

## Description:

An Pong game developed using an OpenAI gym environment to train a Duel Deep Q learning algorithm (DDQN).

# Supported system

Python 3.6+

# Installation:

Make sure you have both git and cmake for this to work.

Cmake: ```https://cmake.org/install/```
Git: ```https://git-scm.com```

# Command lines:

Installing atari-py: ```pip install atari-py```

Installing OpenAI gym SDK: ```pip install gym```

Installing pyTorch: ```pip install torch```

# Importing ROMs

In order to import ROMS, you need to download ```Roms.rar``` from the Atari 2600 VCS ROM Collection and extract the .rar file. Once you've done that, run:
Make a folder named ```ROM``` inside the main project folder and extract what's in ```Roms.rar``` inside that folder.
Then run this command
```python -m atari_py.import_roms <path to folder>```

This should print out the names of ROMs as it imports them. The ROMs will be copied to your installation directory.

# Running the train

You now need to execute the main python file which is ```main_dueling_ddqn.py```.
The train session will start.
