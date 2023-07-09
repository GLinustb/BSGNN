# GraphTTS
GraphTTS is a directed graph-based deep learning model for predicting material properties based on the CrystalNet[1] framework.

In our Model, each node u is represented by an initial feature vector x(u) that collected from the atom fingerprint, each edge 〖(u,v)〗_k is also represented by a raw feature vector x((u,v)_k ), corresponding to the kth bond connecting atom u and v. Note that the metal bonds and the ionic bonds are depended on the distance and the electronegativity between two atoms, we expanded the distance with the Gaussian basis exp⁡(-(r-r_0 )^2/σ^2) centered at 100 points linearly placed between 0 and 5 and σ=0.5.

# Requirement
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

# How to prepare dataset?
Specified the fowllowing files path in proprecess.py

property.csv
cif files
And then run proprecess.py.

# How to run?
python -u train_all.py --seed 4 --data_path ../data/preprocess --train_path ./data/preprocess/calculate --dataset_name band_gap --dataset_type regression --run_fold 0 --metric mae --save_dir ./ckpt/ensemble_band_gap --epochs 100 --init_lr 1e-4 --max_lr 3e-4 --final_lr 1e-4 --no_features_scaling --show_individual_scores --max_num_neighbors 64 > ./log/fold_0_band_gap.log 2>&1 &

# How to predict properties?
python -u predict.py --seed 0 

# Reference:
[1]. Pin Chen, Yu Wang, Hui Yan, Sen Gao, Zexin Xu, Yangzhong Li, Qing Mo, Junkang Huang, Jun Tao, GeChuanqi Pan, Jiahui Li & Yunfei Du. 3DStructGen: an interactive web-based 3D structure generation for non-periodic molecule and crystal. J Cheminform 12, 7 (2020). https://doi.org/10.1186/s13321-020-0411-2
