# MultiAgentTrading
Repository for our MAAI group research project

## Installation
To clone the repository and install the required packages, run the following commands in the terminal:
```
git clone https://github.com/olibridge01/MultiAgentTrading.git
cd MultiAgentTrading
pip install -r requirements.txt
```

## To-Do List
- (**Oli**) Implement trading environment
- (**Oli**) Implement individual DQN agents
- (**Oli**) Implement MADQN algorithm
- (**Shomit?**) Research other algorithms (e.g. PPO, actor-critic)
- (**Shomit**) Data collection (email paper authors)
- (**Isaac**) Rule-based methods (code up)
- Extend env to multi-agent setting
- Early drafting of report? Gathering background theory etc.

## Git Commands

### Initial Setup
To pull the latest changes from the repository, run the following command in the terminal:
```
git pull
```

### Branches
For our development we want to use dev branches. To create a new branch *and switch to it*, run the following command in the terminal:
```
git branch <branchname>
git checkout <branchname>
```

### Committing Changes

To see the changes made to your local repository, run the following command in the terminal:
```
git status
```

To add all changes to the repository, run the following command in the terminal:
```
git add .
```
or to add specific files, run:
```
git add <filename>
```


Now you can make changes to the code and commit them to your branch. To commit the changes to your branch, run the following command in the terminal:
```
git commit -m "Commit message"
```

To push the changes to the repository from your branch, run the following command in the terminal:
```
git push origin <branchname>
```