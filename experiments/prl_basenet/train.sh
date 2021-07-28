#!/bin/bash

#SBATCH --job-name=sseg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=40:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hl3797@nyu.edu
#SBATCH --gres=gpu

module purge
module load anaconda3/2020.07

cd /scratch/$USER/SSeg/experiments/prl_basenet/

source activate dl
python train.py $1 > $1.log 2>&1 &

wait

