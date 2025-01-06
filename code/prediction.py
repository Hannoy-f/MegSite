import torch
import os,sys
import pandas as pd
import esm3_feature
import util
import config
import bulid_protein_graph
import numpy as np
from egnn_clean import *

fastafile=sys.argv[1]
prediction_type=sys.argv[2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pro_dicts=util.fasta2dict(fastafile)
print('1. Esm3 feature generate')
esm3_feature.esm3_feature_generate(fastafile,config.pdb_path,config.tsv_path,config.result_path)

print('2. Load model')
if prediction_type == 'DNA':
    model_path=config.model_path+'DNA-2304-384-3.pth'
elif prediction_type == 'RNA':
    model_path=config.model_path+'RNA-2304-384-3.pth'
else:
    raise ValueError('Prediction type must be DNA or RNA!')
model = EGNN(in_node_nf=2304, hidden_nf=384, out_node_nf=1, in_edge_nf=1, n_layers=3,
             attention=True).to(device)  # 确保模型被移动到正确的设备
model.load_state_dict(torch.load(model_path))
model.eval()

def min_max_normalize(tensor):
    # 计算每个特征的最大值和最小值
    min_values = tensor.min(dim=0).values
    max_values = tensor.max(dim=0).values

    # 避免除以零的情况
    epsilon = 1e-8
    normalized_tensor = (tensor - min_values) / (max_values - min_values + epsilon)

    return normalized_tensor

print('3. Prediction start')
for one_pro in pro_dicts:
    one_graph = bulid_protein_graph.bulid_graph(config.result_path + one_pro+ '.npy', config.pdb_path + one_pro + '.pdb',
                                                config.result_path + one_pro + '.esm_msa', one_pro, one_pro[-1])
    one_graph = one_graph.to(device)
    edge_fea = one_graph.edata['edg_fea']
    edge_fea = edge_fea.unsqueeze(1)
    edge_fea = edge_fea.to(device)
    node_fea = one_graph.ndata['node_fea'].to(torch.float32)
    node_fea = min_max_normalize(node_fea)
    node_fea = node_fea.to(device)
    coord = one_graph.ndata['coord']
    outputs, _ = model(node_fea, coord, one_graph.edges(), edge_fea)
    outputs = torch.nn.Sigmoid()(outputs)

    with open(config.result_path + one_pro + '.txt', 'w') as file:
        for value in outputs:
            file.write(f"{value.item()}\n")  # 将张量元素写入文件，每行一个元素

print('4. Prediction is over!')
