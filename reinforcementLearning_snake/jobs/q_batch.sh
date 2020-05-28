#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=q_learning
#SBATCH --output=q_learning
#SBATCH --mem=800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilse.pubben@gmail.com

module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow

python main.py -a q-learning -e 0.05 -y 0.99 -v 7
