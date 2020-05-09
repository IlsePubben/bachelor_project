#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=other_learning
#SBATCH --output=other_learning
#SBATCH --mem=800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilse.pubben@gmail.com
module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow
for i in {1..5}
do 
    python main.py -a q-learning -e 0.1 -y 0.99 --lrQ 0.0001 --lrV 0.0001
    python main.py -a qv-learning -e 0.1 -y 0.99 --lrQ 0.0001 --lrV 0.0001
    python main.py -a qvmax-learning -e 0.3 -y 0.95 --lrQ 0.0001 --lrV 0.0001
done