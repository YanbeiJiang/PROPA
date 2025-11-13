# PROPA: Toward Process-level Optimization in Visual Reasoning via Reinforcement Learning
Repository for PROPA


## Setup

### Files and Package Required
Run ```pip install vllm``` to install vllm package.

Note that if local swift package is not working under your enviroments, please run ```pip install ms-swift``` to install the latest ms-swift package, and copy the files ```./swift/plugin/orm.py``` and ```./swift/trainers/rlhf_trainer/grpo_trainer.py``` into the same location at new ms-swift package to overwrite the existing file.

Under each dataset folder, 

```SFT``` contains zero-shot and SFT-F baselines

```RL_ORM``` contains RFT-zero baseline
 
```Long_SFT``` contains SFT-CoT and RFT baseline



