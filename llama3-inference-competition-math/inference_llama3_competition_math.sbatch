#!/bin/bash
#SBATCH -Jrtx6000_inference_competition_math             # Job name  
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#SBATCH --gres=gpu:RTX_6000:1              # GPU type (H100) and number of GPUs 

#SBATCH --mem-per-gpu=224GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t15:00:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -ollama3-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=knazarali3@gatech.edu # E-mail address for notifications 

module load anaconda3

source activate base
conda activate 310

cd "/home/hice1/knazarali3/scratch/cs6220_math_group2"

python inference_llama3_competition_math.py