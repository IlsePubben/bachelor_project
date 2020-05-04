#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=qva_learning
#SBATCH --output=qva_learning
#SBATCH --mem=800
module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow
for i in {1..5}
do 
    python main.py -a qva-learning -e 0.1 -y 0.99 --lrA 0.005
    python main.py -a qva-learning -e 0.1 -y 0.99 --lrA 0.0002
done