#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=other_learning
#SBATCH --output=other_learning
#SBATCH --mem=800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilse.pubben@gmail.com
#SBATCH --array=0-3

module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow

algorithm=(q-learning qv-learning qva-learning)
lr=(0.005 0.0005 0.0005)

python main.py -a ${algorithm[${SLURM_ARRAY_TASK_ID}]} -e 0.1 -y 0.99 --lrQ ${lr[${SLURM_ARRAY_TASK_ID}]} --lrV ${lr[${SLURM_ARRAY_TASK_ID}]} --lrA ${lr[${SLURM_ARRAY_TASK_ID}]} -v 5