# Adversarial-Attack-GNN
Machine Learning Project

## Setting Up the Environment

1. **SSH into glogin**:  
    `ssh <pid>@glogin.cs.vt.edu`

2. **Download and create a conda environment**:
   
   `wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh`

   `chmod +x Anaconda3-2023.03-Linux-x86_64.sh`
   
   `./Anaconda3-2023.03-Linux-x86_64.sh`
   
    `source ~/.bashrc`

   Check the version:
   
    `conda --version`

   Create the environment called `huggingface_env`:
   
    `conda create --name huggingface_env python=3.8`
   
    Activate the environment:

   `source activate huggingface_env`

   Install the necessary libraries:
   
    `pip install torch diffusers transformers datasets accelerate`

## Installing the required packages
   
   Right now, based on CPU.

   `pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cpu.html`

   `pip install deeprobust`
   
   `pip install networkx matplotlib`

   For ogbn-proteins dataset:
   
   `pip install ogb`

## Run the Script

First uncomment the dataset(s) you want (Our code runs on Cora, Citeseer, PolBlogs, and PubMed (subgraph)):

```python
if __name__ == "__main__": 
```

Then, run the script:

```bash
python main.py
```

# What you should see

You should have generated a couple of new folders. 

folders:

1. **data** - contains Cora and Citeseer datasets.
2. **tmp** - contains the PolBlogs dataset.
3. **clean_models** - contains clean models on clean dataset.
4. **perturbed_data** - contains the poisoned dataset.
5. **poisoned_models** - contains poisoned models on the poisoned dataset.
6. **visuals** - containing the dataset visuals before and after the attack.
7. **acc_boxplots** - containing the accuracies of the three runs compared to the clean model's accuracy.
8. **results** - containing a `.txt` file outputting the evaluation metrics before and after metattack.

## Specifications

To check which specs you have run these lines in the terminal:

```bash
python -c "import torch; print(torch.__version__)"
conda --version
```

my specs are as followed: 

```bash
2.4.1+cu121
conda 23.1.0
```

     
