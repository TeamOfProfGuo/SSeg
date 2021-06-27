#!/bin/bash
#SBATCH --job-name=GAP_62_ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=44GB
#SBATCH --time=120:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hl3797@nyu.edu # put your email here if you want emails
#SBATCH --output=GAP_62_ef.out
#SBATCH --error=GAP_62_ef.err
#SBATCH --gres=gpu:2 # How much gpu need, n is the number
#SBATCH -p gpu
#SBATCH --constraint=2080Ti

echo "Your NetID is: $1"
echo "Your environment is: $2"

module purge
module load anaconda3
module load cuda/10.0
module load gcc/7.3

cd /gpfsnyu/scratch/$1/SSeg/

# # Comment here if we've added cwd to sys.path
# rm -r /gpfsnyu/home/$1/.conda/envs/$2/lib/python3.6/site-packages/encoding/
# cp -r /gpfsnyu/scratch/$1/DANet/encoding/ /gpfsnyu/home/$1/.conda/envs/$2/lib/python3.6/site-packages/encoding/
# echo "encoding updated"

echo "start training"
source activate $2
python experiments/rn_gap_fuse/train.py
echo "FINISH"