#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=epsilon02
#SBATCH --output=epsilon02
#SBATCH --mem=800
module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow
for i in {1..5}
do 
    python main.py -a qv-learning -e 0.2 -y 0.99
    python main.py -a qvmax-learning -e 0.2 -y 0.99
    python main.py -a qvamax-learning -e 0.2 -y 0.99
done
