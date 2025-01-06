# MegSite
# Environment Setup:
Since ESM2 and ESM3 cannot be installed in the same virtual environment, this experiment requires two virtual environments.

The first virtual environment is used to generate `ems_msa` feature files and requires the following packages:

- HHblits software (https://github.com/soedinglab/hh-suite)
- Uniclust30 database (https://uniclust.mmseqs.com/)
- ESM2 (https://github.com/facebookresearch/esm)
- python==3.10.10,torch==2.3.1,numpy==1.24.3pandas=2.0.3

The second virtual environment is used to generate `esm3` feature files and run the model for prediction:

- ESM3(https://github.com/evolutionaryscale/esm)
- python==3.10.10,torch==2.3.1,numpy==1.24.3,pandas=2.0.3,dssp==2.2.1,dgl==2.0.0


# Set config
The "config.py" file should be set up correctly according to your software environment:

* config.py
 ``` 
pdb_path=r"home/workspace/Master/data/PDB/"
tsv_path=r"home/workspace/Master/data/tsv/"
result_path=r"home/workspace/Master/MegSite/result/"
model_path=r"home/workspace/Master/MegSite/model/"
HHblits=r"home/workspace/Master/software/hhsuite/bin/hhblits"
HHBLITS_DB=r"home/workspace/Master/database/uniclust30_2018_08/uniclust30_2018_08"
 ```

 # Experimental Procedure
- Step 1: Prepare a protein sequence file and save it as `example.fasta`. Then, prepare the corresponding PDB and TSV files and place them in the respective directories specified by `pdb_path` and `tsv_path`.
- Step 2: Activate the first virtual environment and run the following code,The output files from this step will be saved in the directory specified by `result_path`.
 ``` 
 python get_esm_msa.py example.fasta
 ```
- Step 3: Activate the second virtual environment and run the following code,The `prediction_type` should be either `DNA` or `RNA`. The results will be saved as a `.txt` file in the directory specified by `result_path`

```
python prediction.py example.fasta prediction_type
```

# Note
- The `.esm_msa` files, `.a3m` files, `dssp` files, and `.npy` (ESM3) files will be saved in the `result_path` directory.
- Have a nice day!

