#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=qvamax_001_learning
#SBATCH --output=qvamax_001_learning
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
    python main.py -a qvamax-learning -e 0.2 -y 0.95 --lrQ 0.01 --lrV 0.01 --lrA 0.01
    python main.py -a qvamax-learning -e 0.2 -y 0.95 --lrQ 0.01 --lrV 0.01 --lrA 0.05
    python main.py -a qvamax-learning -e 0.2 -y 0.95 --lrQ 0.01 --lrV 0.01 --lrA 0.002
done
