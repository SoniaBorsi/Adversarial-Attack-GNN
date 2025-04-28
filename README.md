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

```bash
python main.py
```

# What you should see

You should have generated a couple of new folders (right now with Cora, Citeseer, PolBlogs, and Texas datasets on CPU). 

1. "data" folder - contains Cora, Citeseer, and Texas datasets.
2. "tmp" folder - contains the PolBlogs dataset.
3. "perturbed_data" folder - containing the saved perturbed model
4. "results" folder - containing a `.txt` file outputting the evaluation metrics before and after metattack.
5. "visuals" folder - containing the dataset visuals before and after the attack.

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

     
