import time
start_time = time.time()
import re
import torch
import argparse
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
from step0_DataPreprocessingSetting import *
import pandas as pd
import os

save_path_npy = path_base + 'redundancy/SPOT1DLM_inputs_new/'
os.makedirs(save_path_npy, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', type=str, help=' define the device you want the ')
args = parser.parse_args()

# 按需加载模型和分词器，避免一次性加载过多模型
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

device = torch.device(args.device)
model = model.to(device)
model = model.eval()

prot_df = pd.read_pickle(path_base + 'pub_data/data_new/new_clean_aa.pkl')

# 设置批处理大小
batch_size = 16
num_batches = len(prot_df) // batch_size + 1

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(prot_df))
    batch_df = prot_df[start_idx:end_idx]

    sequences = batch_df['sequences'].apply(lambda seq: seq.replace('', " "))
    sequences = sequences.apply(lambda seq: re.sub(r"[UZOB]", "X", seq)).tolist()
    prot_names = batch_df['proteins'].tolist()

    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    # 将结果移回CPU并转换为numpy，减少GPU内存占用
    embedding = embedding.last_hidden_state.cpu().numpy()

    for i, prot_name in enumerate(prot_names):
        seq_len = (attention_mask[i] == 1).sum()
        seq_emd = embedding[i][:seq_len - 1]
        np.save(save_path_npy + prot_name + "_pt.npy", seq_emd)

    # 清理不再使用的变量，释放内存
    del input_ids, attention_mask, embedding
    torch.cuda.empty_cache()

print("ProtTrans embeddings generation completed ...")
