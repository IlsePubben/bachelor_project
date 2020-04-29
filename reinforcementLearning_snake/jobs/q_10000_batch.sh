#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=q_10000_e01_y099
#SBATCH --output=q_10000_e01_y099
#SBATCH --mem=800
module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow
for i in {1..5}
do 
    python main.py -a q-learning -e 0.1 -y 0.99
done
