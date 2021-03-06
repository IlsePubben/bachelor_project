#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=qv(a)
#SBATCH --output=qv_a
#SBATCH --mem=800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilse.pubben@gmail.com
#SBATCH --array=0-5

module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow

algorithm=(qv-learning qv-learning qv-learning qva-learning qva-learning qva-learning)
visionGrid=(3 5 7 3 5 7)

python main.py -a ${algorithm[${SLURM_ARRAY_TASK_ID}]} -v ${visionGrid[${SLURM_ARRAY_TASK_ID}]}