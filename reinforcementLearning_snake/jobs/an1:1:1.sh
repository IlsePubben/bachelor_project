#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=an1:1:1
#SBATCH --output=an1:1:1
#SBATCH --mem=800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilse.pubben@gmail.com
#SBATCH --array=0-269

module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow

algorithm=(q-learning qv-learning qva-learning)
visionGrid=(3 3 3 5 5 5 7 7 7)

dir="outputs/experiments/"

if [ ${visionGrid[${SLURM_ARRAY_TASK_ID}%9]} -eq 3 ]
then 
    dir+="3x3/"
fi

if [ ${visionGrid[${SLURM_ARRAY_TASK_ID}%9]} -eq 5 ]
then 
    dir+="5x5/"
fi

if [ ${visionGrid[${SLURM_ARRAY_TASK_ID}%9]} -eq 7 ]
then 
    dir+="7x7/"
fi

dir+="an_1:1:1/"

python main.py -a ${algorithm[${SLURM_ARRAY_TASK_ID}%3]} -v ${visionGrid[${SLURM_ARRAY_TASK_ID}%9]} --name learningRate_annealing_ -d $dir