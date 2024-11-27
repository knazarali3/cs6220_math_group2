#!/bin/bash
#SBATCH -JCS6220llama3Training             # Job name  
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#SBATCH --gres=gpu:V100:1                # GPU type (H100) and number of GPUs 

#SBATCH --mem-per-gpu=224GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t15:00:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -ollama3-training-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=knazarali3@gatech.edu # E-mail address for notifications 

module load anaconda3

source activate base
conda activate 310

cd "/home/hice1/knazarali3/scratch/cs6220_math_group2"

python train_llama3_competition_math.py