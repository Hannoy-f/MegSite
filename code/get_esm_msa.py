import esm_msa_feature
import util
import config
import torch
import os,sys

fastafile=sys.argv[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
util.get_MSA(fastafile,config.result_path,config.HHblits,config.HHBLITS_DB)
pro_dicts=util.fasta2dict(fastafile)
for one_pro in pro_dicts:
    if os.path.exists(config.result_path+one_pro+'.esm_msa'):continue
    esm_msa_feature.generate_data_from_file(config.result_path,one_pro,device)