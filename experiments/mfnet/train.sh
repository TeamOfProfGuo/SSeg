#!/bin/bash

#SBATCH --job-name=DANetd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=88GB
#SBATCH --gres=gpu:aquila
#SBATCH --time=100:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=danet_%j.out
#SBATCH --error=danet_error_%j.out
#SBATCH --gres=gpu:2 # How much gpu need, n is the number
#SBATCH -p aquila
#SBATCH --constraint=2080Ti

module purge
source ~/.bashrc
source activate python36
echo "start training">>train.log
module load cuda/10.0
#module load cudnn/7.5

python train.py >train.log 2>& 1
echo "FINISH"
