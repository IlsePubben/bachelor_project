#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=con1:03:3
#SBATCH --output=con1:03:3
#SBATCH --mem=800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilse.pubben@gmail.com
#SBATCH --array=0-89

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

dir+="con_1:03:3/"

python main.py -a ${algorithm[${SLURM_ARRAY_TASK_ID}%3]} -v ${visionGrid[${SLURM_ARRAY_TASK_ID}%9]} --name con_lrQ:lrV:lrA-1:03:3_ --lrV 0.3333 --lrA 3 --lrBegin 0.005 --lrEnd 0.005 -d $dir