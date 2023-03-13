#!/bin/bash
#SBATCH --job-name=feature_clustering
#SBATCH --output=out_array_%A_%a.out
#SBATCH --error=out_array_%A_%a.err
#SBATCH --array=1-30
#SBATCH --time=6:25:00
#SBATCH --partition=parallel
#SBATCH --mem-per-cpu=4000M
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wangpeng@ecs.vuw.ac.nz

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

file_path=/nfs/scratch/wangpe/ADE/clustering_mine/algorithms
echo $SLURM_JOB_NODELIST

module load python/3.8.1
source /nfs/home/wangpe/python_test/mytest/bin/activate
python $file_path/main_final.py $1 $SLURM_ARRAY_TASK_ID
# echo $right

mv *.txt  /nfs/scratch/wangpe/ADE/clustering_mine/results/p015/$1
mv *.npy  /nfs/scratch/wangpe/ADE/clustering_mine/results/p015/$1
