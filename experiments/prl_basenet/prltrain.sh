#!/bin/bash

#SBATCH --job-name=sseg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hl3797@nyu.edu
#SBATCH --gres=gpu

#GREENE GREENE_GPU_MPS=yes

module purge
module load anaconda3/2020.07

cd /scratch/$USER/SSeg/experiments/prl_basenet/

source activate dl
python train.py $1 > $1.log 2>&1 &
python train.py $2 > $2.log 2>&1 &

wait

