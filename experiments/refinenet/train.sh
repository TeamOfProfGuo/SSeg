#!/bin/bash

#SBATCH --job-name=refinenet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=88GB
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=RefineNet_%j.out
#SBATCH --error=RefineNet_%j.out
#SBATCH --gres=gpu:2 # How much gpu need, n is the number
#SBATCH -p aquila

module purge
source ~/.bashrc
source activate python36
echo "start training">>train.log
module load cuda/10.0
#module load cudnn/7.5

python train.py >train.log 2>& 1
echo "FINISH"

