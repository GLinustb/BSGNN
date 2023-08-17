# bsgnn
bsgnn is a GNN model for predicting superconductivity based on the CrystalNet[1] framework.

In our Model, we add three modules into bsgnn: nearest-neighbors-only graph represent (NGR), communicative message passing (CMP) and attention (GAT) module. First, CGR module represents the ordered and disordered crystal structures as periodic graphs. Specifically, each node is represented by an initial feature vector that collected from the atom fingerprint, each edge is also represented by a raw feature vector, corresponding to the bond connecting two atoms. Here, only the nearest neighboring nodes were connected with central nodes and lattice distortion were taken into consideration, which allows the graph represents crystal structure correctly. Second, CMP module mimics the complex physical and chemical interactions between atoms and bonds. where the message interactions were strengthened between atoms and bonds through communicative message passing. Third, attention module give different weights to neighboring nodes during message passing. 


# Requirement
```
numpy                 1.20.2
pandas                1.2.4
pymatgen              2020.12.18
pyparsing             2.4.7
scikit-learn          0.24.1
scipy                 1.6.3
torch                 1.5.0+cu101
torch-cluster         1.5.5
torch-geometric       1.5.0
torch-scatter         2.0.5
torch-sparse          0.6.6
torch-spline-conv     1.2.0
torchaudio            0.5.0
torchvision           0.6.0+cu101
tornado               6.1
tqdm                  4.60.0
```
# How to prepare dataset?
Specified the fowllowing files path in proprecess.py
- cif_files.csv
- data_cif


And then run proprecess.py.

# How to run?
```
python -u train_all.py --seed 4 --data_path ./data --dataset_type regression --metric r2 --save_dir ./test --epochs 100 --init_lr 1e-4 --max_lr 3e-4 --final_lr 1e-4 --no_features_scaling --show_individual_scores --max_num_neighbors 64
```
# How to predict properties?
```
python -u predict.py --test_path ./data/predict.csv --checkpoint_dir ./test --preds_path ./result/predict_result.csv
```
# Reference:
[1]. Chen P, Chen J, Yan H, et al. Improving Material Property Prediction by Leveraging the Large-Scale Computational Database and Deep Learning[J]. The Journal of Physical Chemistry C, 2022, 126(38): 16297-16305.
