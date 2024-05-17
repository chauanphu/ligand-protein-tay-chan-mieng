from rdkit import Chem
import os
import pandas as pd
import numpy as np

from utils.classes import Feature_extractor

def get_min(compound_positions):
    """
    Returns the minimum x, y, and z coordinates from a list of compound positions.

    Args:
        compound_positions (list): A list of compound positions, where each position is a tuple of x, y, and z coordinates.

    Returns:
        tuple: A tuple containing the minimum x, y, and z coordinates.

    Example:
        >>> positions = [(1, 2, 3), (4, 5, 6), (0, 2, 1)]
        >>> get_min(positions)
        (0, 2, 1)
    """
    minx, miny, minz = 999, 999, 999
    for pos in compound_positions:
        x, y, z = pos
        if x < minx:
            minx = x

        if y < miny:
            miny = y

        if z < minz:
            minz = z

    return (minx,miny,minz)

def adjust_grid(compound, compound_positions, protein_path, isprotein, grid, minx, miny, minz):
    """
    Adjusts the grid by adding features to specific grid positions based on the compound and protein information.

    Args:
        - compound: The compound object.
        - compound_positions: The positions of the compound atoms.
        - protein_path: The path to the protein file.
        - isprotein: A boolean indicating whether the compound is a protein.
        - grid: The grid to be adjusted.
        - minx: The minimum x-coordinate value.
        - miny: The minimum y-coordinate value.
        - minz: The minimum z-coordinate value.

    Returns:
        The adjusted grid.
    """
    atoms_aa = []
    with open(protein_path, 'r+') as f:
        readlines = f.readlines()
        f.close()

    for idx, lines in enumerate(readlines):
        if 'HETATM' in lines or 'ATOM' in lines:
            atoms_aa.append(lines[17:20])

    for idx, pos in enumerate(compound_positions):
        x, y, z = pos

        amino_acid = atoms_aa[idx]
        atom = compound.GetAtomWithIdx(int(idx))
        features = get_atom_features(atom, amino_acid, isprotein)
        features.extend(pos)
        
        x_rounded = round(x - minx)
        y_rounded = round(y - miny)
        z_rounded = round(z - minz)

        if grid[x_rounded, y_rounded, z_rounded][0] == 0:
            grid[x_rounded, y_rounded, z_rounded] = features
        elif grid[x_rounded+1, y_rounded, z_rounded][0] == 0:
            grid[x_rounded+1, y_rounded, z_rounded] = features
        elif grid[x_rounded, y_rounded+1, z_rounded][0] == 0:
            grid[x_rounded, y_rounded+1, z_rounded] = features
        elif grid[x_rounded, y_rounded, z_rounded+1][0] == 0:
            grid[x_rounded, y_rounded, z_rounded+1] = features
        elif grid[x_rounded+1, y_rounded+1, z_rounded][0] == 0:
            grid[x_rounded+1, y_rounded+1, z_rounded] = features
        elif grid[x_rounded, y_rounded+1, z_rounded+1][0] == 0:
            grid[x_rounded, y_rounded+1, z_rounded+1] = features

    return grid

def add_ligand(ligand_path, grid, minx, miny, minz, isprotein = 0):
    ligand = Chem.MolFromPDBFile(ligand_path, False, False, 1)
    ligand_conf = ligand.GetConformer()
    ligand_positions = ligand_conf.GetPositions()

    result = adjust_grid(ligand, ligand_positions, ligand_path, isprotein, grid, minx, miny, minz)

    return result

def get_batch_list(dataset_idx: list, batch_size: int, index: int):
    '''
    Get the list of indices for the current batch

    Args:
        dataset_idx (list): The list of indices for the dataset.
        batch_size (int): The size of the batch.
        index (int): The index of the current batch.
    '''
    # Get the list of indices for the current batch
    return [value for idx, value in enumerate(dataset_idx) if idx >= index * batch_size and idx < (index+1)*batch_size]

def get_grid(protein_path: str, complexFile_path: str):
    '''
    Create a grid for the protein and ligand complex.

    Args:
        - protein_path (str): The path to the protein file.
        - complexFile_path (str): The path to the complex file.
    '''
    grid, minx, miny, minz = set_grid(protein_path)
    return add_ligand(complexFile_path, grid, minx, miny, minz)

def get_label(label_list: list, from_protein: str, complex_name: str, label_folder: str):
    protein = [value for value in label_list if from_protein in value][0]
    label_file_path = os.path.join(label_folder, protein)
    df = pd.read_csv(label_file_path)
    listidx = df.index[df['file.pdb'] == complex_name].tolist()[0]
    ba = df['BA'][listidx]
    stat = df['Hit/No_hit'][listidx]
    if stat == 'hit':
        stat = 1
    else:
        stat = 0
    return ba, stat

def set_grid(protein_path, isprotein= 1):
    # Load the protein file
    compound = Chem.MolFromPDBFile(protein_path, False, False, 1)
    # Get the conformer and positions of the compound
    compound_conf = compound.GetConformer()
    compound_positions = compound_conf.GetPositions()
    # Get the first atom of the compound
    atom = compound.GetAtomWithIdx(int(1))
    features = get_atom_features(atom, '', isprotein)

    minx, miny, minz = get_min(compound_positions)
    # Generate empty grid with size 52x52x52x(features+3)
    grid=np.zeros((52,52,52,len(features)+3))

    adjusted_grid = adjust_grid(compound, compound_positions, protein_path, isprotein, grid, minx, miny, minz)

    return adjusted_grid, minx, miny, minz

def get_file_lists(protein_folder, ligand_folder, label_folder):
    ligand_list = os.listdir(ligand_folder)
    protein_list = os.listdir(protein_folder)
    label_list = os.listdir(label_folder)
    return ligand_list, protein_list, label_list

def get_complex_info(complexFile):
    from_protein = complexFile.split('-')[0]
    complex_name = complexFile.split('.')[0]
    return from_protein, complex_name

def get_data_batch(
    dataset_idx: list, 
    protein_folder: str, 
    ligand_folder: str, 
    ligand_list: list, 
    label_folder: str, 
    label_list: list, 
    batch_size: int, 
    index: int):
    '''
    Get the data for the current batch

    Args:
        - dataset_idx (list): The list of indices for the dataset.
        - protein_folder (str): The path to the protein folder.
        - ligand_folder (str): The path to the ligand folder.
        - ligand_list (list): The list of ligand files. Ex: 3qzq-8v_model1.pdb
        - label_folder (str): The path to the label folder.
        - label_list (list): The list of label files. Ex: 3qzq systems.csv
        - batch_size (int): The size of the batch.
        - index (int): The index of the current batch.
    '''
    baList = []
    statList = []
    gridList = []

    batch_list = get_batch_list(dataset_idx, batch_size, index)
  
    for i in batch_list:
        # Complex file: 3qzq-8v_model1.pdb
        complexFile = ligand_list[i]
        # From protein: 3qzq
        from_protein = complexFile.split('-')[0]
        # Complex name: 3qzq-8v_model1
        complex_name = complexFile.split('.')[0]
        protein_path = os.path.join(protein_folder, from_protein + '.pdb')
        complexFile_path = os.path.join(ligand_folder, complexFile)
        grid = get_grid(protein_path, complexFile_path)
        gridList.append(grid)
    
        ba, stat = get_label(label_list, from_protein, complex_name, label_folder)
        baList.append(ba)
        statList.append(stat)
  
    gridList = np.array(gridList)
    baList = np.array(baList)
    statList = np.array(statList)

    return gridList, baList, statList

def get_atom_features(atom: Chem.Atom, amino_acid: str, isprotein):
    '''
    Get the features of an atom

    Args:
        - atom (Chem.Atom): The atom object.
        - amino_acid (str): The amino acid of the atom.
        - isprotein (int): A boolean indicating whether the atom is a protein.

    Returns:
        - list: A list of atom features:
            - classes (int): The class of the atom. Ex: 0 for B, 1 for C, 2 for N, 3 for O, 4 for P, 5 for S, 6 for Se, 7 for halogen, 8 for metal, 9 for others.
            - chirality (int): The chirality of the atom. Ex: 0 for CHI_UNSPECIFIED, 1 for CHI_TETRAHEDRAL_CW, 2 for CHI_TETRAHEDRAL_CCW, 3 for CHI_OTHER.
            - charge (int): The charge of the atom.
            - hyb (int): The hybridization of the atom.
            - numH (int): The number of hydrogen atoms.
            - valence (int): The valence of the atom.
            - degree (int): The degree of the atom.
            - aromatic (int): A boolean indicating whether the atom is aromatic.
            - mass (float): The mass of the atom.
            - amino_acid (int): The amino acid of the atom.
            - isprotein (int): A boolean indicating whether the atom is a protein.

    '''
    ATOM_CODES = {}
    metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
              + list(range(37, 51)) + list(range(55, 84))
              + list(range(87, 104)))
    atom_classes = [(5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (15, 'P'), (16, 'S'), (34, 'Se'),
                    ([9, 17, 35, 53], 'halogen'), (metals, 'metal')]
    for code, (atomidx, name) in enumerate(atom_classes):
        if type(atomidx) is list:
            for a in atomidx:
                ATOM_CODES[a] = code
        else:
            ATOM_CODES[atomidx] = code
    try:
        classes = ATOM_CODES[atom.GetAtomicNum()]
    except:
        classes = 9

    possible_chirality_list = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
    chirality = possible_chirality_list.index(atom.GetChiralTag())

    possible_formal_charge_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    try:
        charge = possible_formal_charge_list.index(atom.GetFormalCharge())
    except:
        charge = 11

    possible_hybridization_list = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]
    try:
        hyb = possible_hybridization_list.index(atom.GetHybridization())
    except:
        hyb = 6

    possible_numH_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    try:
        numH = possible_numH_list.index(atom.GetTotalNumHs())
    except:
        numH = 9

    possible_implicit_valence_list = [0, 1, 2, 3, 4, 5, 6, 7]
    try:
        valence = possible_implicit_valence_list.index(atom.GetTotalValence())
    except:
        valence = 8

    possible_degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    try:
        degree = possible_degree_list.index(atom.GetTotalDegree())
    except:
        degree = 11

    is_aromatic = [False, True]
    aromatic = is_aromatic.index(atom.GetIsAromatic())

    mass = atom.GetMass() / 100

    # idx = atom.GetIdx()
    # with open(protein_path, 'r+') as f:
    #     readlines = f.readlines()
    #     f.close()

    amino_acids = [
        'ALA', 'ARG', 'ASN', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]
    if amino_acid in amino_acids:
      amino_acid = amino_acids.index(amino_acid)
    else:
      amino_acid = int(len(amino_acids) + 1)

    # amino_acid = amino_acids.index(amino_acid)
    # amino_acid = 0
    # for lines in readlines:
    #     if 'HETATM' in lines or 'ATOM' in lines:
    #         if idx == int(lines[6:11]):
    #             amino_acid = lines[17:20]
                # if amino_acid in amino_acids:
                #     amino_acid = amino_acids.index(amino_acid)
                # else:
                #     amino_acid = int(len(amino_acids) + 1)

    return [classes, chirality, charge, hyb, numH, valence, degree, aromatic, mass, amino_acid, isprotein]
    core_grids=None
    core_ba= []
    ligand_list = os.listdir(ligand_folder)
    batch_list = [value for idx, value in enumerate(dataset_idx) if idx >= index * batch_size and idx < (index+1)*batch_size ]
    for i in batch_list:
        complexFile = ligand_list[i]
        from_protein = complexFile.split('-')[0]
        complex_name = complexFile.split('.')[0]
        # Load the data specific to 3DCNN model
        ligand_train_path = os.path.join(ligand_folder, complexFile)
        protein_train_path = os.path.join(protein_folder, from_protein+'.pdb')
        label_train_path = os.path.join(label_folder, complex_name+'.txt')
        ligand = np.load(ligand_train_path)
        protein = np.load(protein_train_path)
        label = np.loadtxt(label_train_path)
        if core_grids is None:
            core_grids = np.concatenate((protein, ligand), axis=-1)
            core_ba = label
        else:
            core_grids = np.concatenate((core_grids, np.concatenate((protein, ligand), axis=-1)), axis=0)
            core_ba = np.concatenate((core_ba, label), axis=0)
    return core_grids, core_ba