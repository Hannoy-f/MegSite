import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import subprocess

from esm.pretrained import (
    ESM3_sm_open_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization import get_model_tokenizers
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.utils.structure.protein_chain import ProteinChain
from Bio.PDB import PDBParser, SASA
from esm.utils.types import FunctionAnnotation


tokenizers = get_model_tokenizers()
model = ESM3_sm_open_v0('cuda' if torch.cuda.is_available() else 'cpu')
encoder = ESM3_structure_encoder_v0('cuda' if torch.cuda.is_available() else 'cpu')

def get_sequence_tokens(sequence):
    tokens = tokenizers.sequence.encode(sequence)
    sequence_tokens = torch.tensor(tokens, dtype=torch.int64)
    sequence_tokens = sequence_tokens.unsqueeze(0)
    print("sequence_tokens:", sequence_tokens.size())
    return sequence_tokens

def get_structure_tokens(pdbfile):
    chain = ProteinChain.from_pdb(pdbfile)
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097
    print("structure_tokens:", structure_tokens.size())
    return structure_tokens

def get_sasa_tokens(pdbfile):
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("example",pdbfile)
    sasa_calculator = SASA.ShrakeRupley()
    sasa_calculator.compute(structure, level="R")
    sasa_list = []
    for modelx in structure:
        for chain in modelx:
            for residue in chain:
                if residue.id[0] == " ":
                    sasa_list.append(residue.sasa)
    sasa_tokens = tokenizers.sasa.encode(sasa_list)
    sasa_tokens = sasa_tokens.unsqueeze(0)
    print("sasa_tokens:", sasa_tokens.size())
    return sasa_tokens

def get_ss8_tokens(PDB_ID,pdbfile,result_path):
    dssp_name=PDB_ID+'.dssp'
    dssp_path= os.path.join(result_path, dssp_name)
    subprocess.run(['mkdssp', '-i', pdbfile, '-o', dssp_path], check=True)
    with open(dssp_path, 'r') as file:
        a = file.readlines()[28:]
    stucture = 16
    stc = ""
    for liness in a:
        if '!' in liness:
            continue
        if liness[13] == "X":
            continue
        stc += liness[stucture]
    ss8 = stc.replace(" ", 'C')
    ss8_tokens = tokenizers.secondary_structure.encode(ss8)
    ss8_tokens = ss8_tokens.unsqueeze(0)
    file.close()
    print("ss8_tokens:", ss8_tokens.size())
    #ss8_tokens=None
    return ss8_tokens

def get_function_tokens(tsvfile,sequence):
    if os.path.exists(tsvfile):
        function_annotations = []
        with open(tsvfile, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                # print(parts)
                label = parts[2]
                start = int(parts[0])
                end = int(parts[1])
                annotation = FunctionAnnotation(label, start, end)
                function_annotations.append(annotation)
        function_tokens = tokenizers.function.tokenize(function_annotations, len(sequence))
        function_tokens = tokenizers.function.encode(function_tokens)
        print("function_tokens:", function_tokens.size())
    else:
        function_tokens = None
        print("function_tokens:None")
    return function_tokens

def esm3_feature_generate(fastafile,pdb_path,tsv_path,result_path):
    with open(fastafile, 'r') as txtFile:
        for n, line in enumerate(txtFile):
            if (n + 1) % 2 == 1:
                PDB_ID = line.strip()[1:]
            if (n + 1) % 2 == 0:
                sequence = line.strip()
                sequence_tokens = get_sequence_tokens(sequence)
                structure_tokens = get_structure_tokens(pdb_path+PDB_ID+".pdb")
                sasa_tokens = get_sasa_tokens(pdb_path+PDB_ID+".pdb")
                ss8_tokens = get_ss8_tokens(PDB_ID,pdb_path+PDB_ID+".pdb",result_path)
                function_tokens = get_function_tokens(tsv_path+PDB_ID+".tsv", sequence)

                output = model.forward(structure_tokens=structure_tokens, sasa_tokens=sasa_tokens,
                                       sequence_tokens=sequence_tokens, ss8_tokens=ss8_tokens,
                                       function_tokens=function_tokens)

                array = output.embeddings.data
                array = np.squeeze(array)
                esm3_feature = array[1:-1, :]
                np.save(result_path + PDB_ID + ".npy", esm3_feature)




