# cs6220_math_group2

To setup a pace environment to fine-tune the models:
1. Connect to the GT VPN
    - https://vpn.gatech.edu/global-protect/login.esp
2. ssh in terminal of choice:
    - 'ssh gtusername@login-pace.ice.gatech.edu'
    - gtusername = gburdell2, etc.
3. Create a workspace for the project
    - 'mkdir cs6220_project'
    - Could also git clone this repository directly (reference https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls for           authentication when cloning the repository)
4. Create an anaconda environment
    - 'module load anaconda3'
    - 'conda create -n cs6220'
    - 'conda activate cs6220'
5. Install dependencies (check that dependency was installed via 'conda list')
    - torch: 'conda install pytorch torchvision cudatoolkit -c python'
    - transformers: 'conda install conda-forge::transformers'
    - trl: 'pip install trl'
    - datasets: 'conda install conda-forge::datasets'
    - bitsandbytes: 'pip install -U bitsandbytes'
    - einops, timm: 'pip install timm einops'

SAT Math Dataset from HuggingFace; We only take questions with correct solutions.
80% Train 10% Test 10% Validation

To load dataset,
```
from datasets import load_from_disk

sat_math_datasets_splits = load_from_disk('sat-math-datasets-splits')
print(sat_math_datasets_splits)
```

```
DatasetDict({
    train: Dataset({
        features: ['id', 'question', 'reasoning_chain', 'answer', 'is_correct'], 
        num_rows: 29244
    }) 
    test: Dataset({ 
        features: ['id', 'question', 'reasoning_chain', 'answer', 'is_correct'],
        num_rows: 1625
    }) 
    valid: Dataset({
        features: ['id', 'question', 'reasoning_chain', 'answer', 'is_correct'],  
        num_rows: 1625
    }) 
})
```
