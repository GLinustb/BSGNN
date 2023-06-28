from argparse import Namespace
from typing import List, Tuple, Union
import json
import os
import re
import pandas as pd
from pymatgen.util.coord import get_angle
from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN,EconNN
from pymatgen.core import Structure
from rdkit import Chem
import torch
import numpy as np
import math

    # Memoization
ATOM_ID = {}
SMILES_TO_GRAPH = {}
ATOM_FDIM = 16
BOND_FDIM = 104




def get_atom_radius(site):
    # import numpy as np
    # from pymatgen.core.periodic_table import Element


    atom_radius = [Element[e.name].data["Van der waals radius"]
                  for e in site.species.element_composition.elements]
    occupancy = list(site.species.element_composition._data.values())

    # # ---
    wavg_atom_radius = np.dot(atom_radius, occupancy)

    return wavg_atom_radius

def get_cutoff(structure):
    radii = np.array([get_atom_radius(site) for site in structure])
    cutoff = radii[:,np.newaxis] + radii[np.newaxis, :]
    vol_atom = (4 * np.pi / 3) * np.array([r**3 for r in radii]).sum()
    factor_vol = (structure.volume / vol_atom)**(1.0/3.0)
    factor = factor_vol*1
    cutoff *= factor
    return cutoff

def _get_close_neighbors(structure, d_threshold=0.0):
    """
    Get bonds in crystal.
    "bond" means the distance of two neighboring atoms are close enough.
    Data of "Van der Waals radius" are required.

    # Van der Waals radius for the element.
    # This is the empirical value determined from critical reviews of X-ray diffraction, gas kinetic collision cross-section, and other experimental data by Bondi and later workers.
    # The uncertainty in these values is on the order of 0.1 Å.
    # Data are obtained from
    # “Atomic Radii of the Elements” in CRC Handbook of Chemistry and Physics,
    # 91st Ed.; Haynes, W.M., Ed.; CRC Press: Boca Raton, FL, 2010.
    # https://pymatgen.org/pymatgen.core.periodic_table.html

    Parameters
    ----------
    structure : object (pymatgen.core.structure.Structure)
        derived from cif by pymatgen.

    d_threshold : float
        The criterion is ((d - r1 - r2) < d_threshold),
        where r1 and r2 are "Van der Waals radius".
        default is 0.0 .

    Returns
    -------
    close_neighbors : dict
        {
        "center_atom": object (pymatgen.core.sites.PeriodicSite),
        "center_index": int32,
        "center_is_ordered": bool,
        "center_element": dict,
        "center_frac_coords": Array of float64,
        "center_vdW_radius": float64,  (weighted average value when is_ordered == False)
        "neighbor_atom": object (pymatgen.core.sites.PeriodicSite),
        "neighbor_index": neighbor_index,
        "neighbor_is_ordered": bool,
        "neighbor_element": dict,
        "neighbor_frac_coords": Array of float64,
        "neighbor_vdW_radius": float64,  (weighted average value when is_ordered == False)
        "offset_vector": Array of float64,
        "distance": float64
        }
    """
    _center_indices, _points_indices, _offset_vectors, _distances = [],[],[],[]

    dict_threshold = {}

    all_sites = structure.sites
    close_neighbors = []
    _center_indices, _points_indices, _offset_vectors, _distances = [],[],[],[]
    # cutoff = get_cutoff(structure)
    center_indices, neighbor_indices, offset_vectors, distances = structure.get_neighbor_list(7)
    df_center_indices = pd.DataFrame(center_indices)
    df_neighbor_indices = pd.DataFrame(neighbor_indices)
    df_offset_vectors = pd.DataFrame(offset_vectors)
    df_distances = pd.DataFrame(distances)
    cpod = pd.concat([df_center_indices,df_neighbor_indices,df_offset_vectors,df_distances],axis=1)

    for atom in range(max(list(center_indices))+1):
        h = cpod[cpod.iloc[:,0] == atom]
        all_distance = [round(d,2) for d in list(h.iloc[:,5])]
        all_distance.append(100)
        distance_min = np.sort(list(set(all_distance)))[0]
        # distance_2min = np.sort(list(set(all_distance)))[1]
        dict_threshold[atom] = distance_min
    for i in range(len(center_indices)):
        center_index, neighbor_index, offset_vector, distance = center_indices[i], neighbor_indices[i], offset_vectors[i], distances[i]
        if distance < max(dict_threshold[center_index], dict_threshold[neighbor_index]) * 1.2:
                _center_indices.append(center_index)
                _points_indices.append(neighbor_index)
                _offset_vectors.append(offset_vector)
                _distances.append(distance)
    return _center_indices, _points_indices, _offset_vectors, _distances




class AtomInitializer(object):
    """
    Base class for initializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_features(self, site):
        site_feature = []
        for k,v in site.items():
            el = k.number
            feature = self._embedding[el]
            site_feature.append(np.array(feature) * v)

        site_feature = sum(site_feature)

        return site_feature

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self):
        # 92 dimensions
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.ru
    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------

        distances: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


def load_radius_dict(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
    lines = [line.replace(' ', '').strip('\n') for line in lines][1:-1]
    return {item.split(':')[0]: np.float(item.split(':')[1]) for item in lines}


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self,
                 smiles: str,
                 crystals: dict,
                 args: Namespace,
                 radius=7, dmin=0, dmax=5, step=0.05, var=0.5
                 ):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.crystals = Structure.from_dict(crystals).get_primitive_structure()
        self.ari = AtomCustomJSONInitializer(os.path.join('D:\GraphTTS', 'atom_init.json'))
        self.gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step, var=var)
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.bonds = []

        # Convert smiles to molecule
        # mol = Chem.MolFromSmiles(smiles)

        # fake the number of "atoms" if we are collapsing substructures
        # self.n_atoms = mol.GetNumAtoms()
        self.n_atoms = len(self.crystals)

        # Get atom features
        # for i, atom in enumerate(mol.GetAtoms()):
        #     self.f_atoms.append(atom_features(atom))
        # self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
        self.f_atoms = np.vstack([self.ari.get_atom_features(self.crystals[i].species._data)
                                  for i in range(len(self.crystals))])

        center_indices, points_indices, offset_vectors, distances = _get_close_neighbors(self.crystals)
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1, a2, offset, distance in zip(center_indices, points_indices, offset_vectors, distances):
            f_bond = self.gdf.expand(distance).flatten()
            if args.atom_messages:
                self.f_bonds.append(self.f_atoms[a1].tolist() + f_bond.tolist() + offset.tolist())
                self.f_bonds.append(self.f_atoms[a2].tolist() + f_bond.tolist() + offset.tolist())
            else:
                self.f_bonds.append(f_bond.tolist() + offset.tolist())
                self.f_bonds.append(f_bond.tolist() + offset.tolist())

            # Update index mappings
            b1 = self.n_bonds
            b2 = b1 + 1
            self.a2b[a2].append(b1)  # b1 = a1 --> a2
            self.b2a.append(a1)
            self.a2b[a1].append(b2)  # b2 = a2 --> a1
            self.b2a.append(a2)
            self.b2revb.append(b2)
            self.b2revb.append(b1)
            self.n_bonds += 2
            self.bonds.append(np.array([a1, a2]))


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + args.atom_messages * self.atom_fdim  # * 2

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1   # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1   # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]                        # mapping from atom index to incoming bond indices
        b2a = [0]                         # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]                      # mapping from bond index to the index of the reverse bond
        bonds = [[0, 0]]

        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])  # if b!=-1 else 0

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1], self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        bonds = np.array(bonds).transpose(1, 0)

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.bonds

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch: List[str],
              crystals_batch: List[dict],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles, crystals in zip(smiles_batch, crystals_batch):

        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, crystals, args)
            if not args.no_cache and len(SMILES_TO_GRAPH) <= 20000:
                SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, args)
