#!/bin/bash
#SBATCH --job-name=pdl_lrm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=11GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hl3797@nyu.edu # put your email here if you want emails
#SBATCH --output=pdl_lrm.out
#SBATCH --error=pdl_lrm.err
#SBATCH --gres=gpu:1 # How much gpu need, n is the number

echo "Your NetID is: $1"
echo "Your environment is: $2"

module purge
module load anaconda3/2020.07
# module load cuda/10.2.89

cd /scratch/$1/SSeg/experiments/basenet/

echo "start training"
source activate $2
python train.py pdl_lrm
echo "FINISH"
