import pandas as pd
import os
import shutil
from tqdm import tqdm
import pickle
import math
import re
from os import listdir
import numpy as np
import pandas as pd
import random
import os
from shutil import copyfile
from pymatgen.core import Structure

if os.path.exists(r'./data/graph_cache.pickle'):
    with open(r'./data/graph_cache.pickle', 'rb') as f:
        all_data = pickle.load(f)
else:
    all_data = {}

df = pd.read_csv('./data/cif_files.csv', index_col=0)
for crystal_name in df.index:
    all_data[crystal_name] = Structure.from_file(os.path.join(r'./data_cif', crystal_name)).as_dict()
with open(r'./data/graph_cache.pickle', 'wb') as f:
    pickle.dump(all_data, f)

