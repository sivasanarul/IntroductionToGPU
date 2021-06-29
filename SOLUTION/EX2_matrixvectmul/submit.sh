#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res1.txt

#SBATCH --ntasks=1
#SBATCH --time=05:00

#SBATCH --partition=gpu
#SBATCH --nodelist=gpu01

module purge                       
module load CUDA

# Operations
echo "Job start"

./matvec_onethread


# Operations
echo "Job end"
