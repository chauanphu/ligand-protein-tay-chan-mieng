# Hit-to-Lead Prediction from Identifying Docking Sites in Protein-Ligand Complexes With 3D Neural Network

## Table of Content:

- [Folder Structure](#folder-structure)
- [Folder Structure](#folder-structure)

## Folder structure

The folder system, which takes folder "Code" as `src` consists of:

### Jupyter Notebook

- 3DVisualization.ipynb: The Jupyter Notebook for training new model
- RunThesisModels.ipynb: 

### Input folders

- "label" folder: containing labels files: such as "3qzq systems.csv"
- "protein" folder: containing .pdb files about proteins, such as "3qzq.pdb". The protein .pdb files can be extracted from ligand files.
- "ligand" folder: containg .pdb files for interactions between proteins and ligands, such as "3qzq-8v_model1.pdb".

## Output folders

From 3DVisualization.ipynb

- SFCNN/Best1/Best.keras: The best performance model after training.
- SFCNN/Model1/Model1.keras: the model after trainining with the data.
- Final/CSV/resultSFCNN.csv

From RunThesisModels.ipynb

- SFCNN/Save
- SFCNN/Best
- 3DCNN/Save
- 3DCNN/Best
- test/Save
- test/Best