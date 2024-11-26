#!/bin/bash
#SBATCH -H100_inference_sat_math             # Job name  
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#SBATCH --gres=gpu:H100:1              # GPU type (H100) and number of GPUs 

#SBATCH --mem-per-gpu=224GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t15:00:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -oQWEN-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=mhu94@gatech.edu # E-mail address for notifications 

module load anaconda3

source activate base
conda activate cs6220


python inference_QWEN_sat_math2.py