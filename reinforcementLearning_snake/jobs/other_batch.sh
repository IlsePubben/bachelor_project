#!/bin/bash
#SBATCH --time=20:00:00
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
for i in {1..1}
do 
    python main.py -a q-learning -e 0.1 -y 0.99 --lrQ 0.005 --lrV 0.005 -v 5
    python main.py -a qv-learning -e 0.1 -y 0.99 --lrQ 0.0005 --lrV 0.0005 -v 5
    python main.py -a qva-learning -e 0.1 -y 0.99 --lrQ 0.0005 --lrV 0.0005 -v 5
done