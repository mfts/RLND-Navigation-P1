
  

[//]: #  (Image References)

  

[image1]: ./img/trained.gif  "Trained Agent"

[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png  "Kernel"

  

# Training an agent to navigate

  

## Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

  

![Trained Agent][image1]

  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

-  **`0`** - move forward.

-  **`1`** - move backward.

-  **`2`** - turn left.

-  **`3`** - turn right.

  

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

  

## Setup

### Installation

This project contains a Jupyter notebook and several Python files.

  

To set up your python environment to run the code in this repository, follow the instructions below.

  

1. Create (and activate) a new environment with Python 3.6.

	```

	conda create --name drlnd python=3.6
	source activate drlnd

	```

2. Clone repository and install python dependencies

	```

	git clone REPO_NAME
	cd REPO_NAME
	pip install -r requirements.txt

	```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.

	```bash
	python -m ipykernel install --user --name drlnd --display-name "drlnd"

	```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

	![Kernel][image2]

  

### Set up Unity-ML Agents Environment

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

  

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

  

2. Place the file in the root of the Github repository's folder (`REPO_NAME`), and unzip (or decompress) the file.

  
  

## Instructions

  

Navigate to `Navigation.ipynb`. You have the options either to

  

- a) train your own agent based on my model

- b) see my trained agent in action

  

For a), please execute 1-5.

For b), please execute 1-4 and 6.

####  For improvments to the agent or replay of improved agents, please modify the following files:
- **Dueling DQN**:
	``` 
	agent.py:3
	# before
	from model import QNetwork as QNetwork
	# after
	from model import DuelQNetwork as QNetwork
	```
- **Double DQN**:
	```
	agent.py:63
	# before
	self.learn(experiences, GAMMA)
	# after
	self.doublelearn(experiences, GAMMA)
	```
- **Dueling Double DQN**: change both lines as instructed above