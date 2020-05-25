#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=all_algorithms
#SBATCH --output=all_algorithms
#SBATCH --mem=800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilse.pubben@gmail.com
#SBATCH --array=0-17

module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow

algorithm=(q-learning qv-learning qva-learning)
visionGrid=(3 3 3 5 5 5 7 7 7)

epsilon=0.05
if [ ${SLURM_ARRAY_TASK_ID} -gt 8 ]
then 
    epsilon=0.1
fi 

python main.py -a ${algorithm[${SLURM_ARRAY_TASK_ID}%3]} -e ${epsilon} -v ${visionGrid[${SLURM_ARRAY_TASK_ID}]}