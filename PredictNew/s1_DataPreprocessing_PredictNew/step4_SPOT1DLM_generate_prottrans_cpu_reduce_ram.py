import time
start_time = time.time()
import re
import torch
import argparse
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
from step0_DataPreprocessingSetting import *
import pandas as pd
import gc  # 添加垃圾回收模块

save_path_npy = path_base + 'redundancy/SPOT1DLM_inputs_new/'
os.system('mkdir -p %s' % save_path_npy)

parser = argparse.ArgumentParser()
# 使用CPU运算
parser.add_argument('--device', default='cpu', type=str, help='define the device you want to use')  # 修改为默认使用CPU
args = parser.parse_args()

# ### original:
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

# # 使用本地路径加载模型和tokenizer
# tokenizer = T5Tokenizer.from_pretrained(path_Prot_T5_XL_UniRef50, do_lower_case=False)
# model = T5EncoderModel.from_pretrained(path_Prot_T5_XL_UniRef50)

device = torch.device(args.device)
model = model.to(device)
model = model.eval()

prot_df = pd.read_pickle(path_base + 'pub_data/data_new/new_clean_aa.pkl')

k = 0
batch_size = 100  # 添加批处理大小以减少内存使用
num_batches = len(prot_df) // batch_size + int(len(prot_df) % batch_size != 0)

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(prot_df))
    batch_df = prot_df.iloc[start_idx:end_idx]
    print(f"Processing batch {i + 1}/{num_batches}")

    for index, row in batch_df.iterrows():
        print(k)
        k += 1

        seq = row['sequences']
        prot_name = row['proteins']

        seq_temp = seq.replace('', " ")
        sequences_Example = [seq_temp]
        sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
        ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        embedding = embedding.last_hidden_state.cpu().numpy()  # 确保嵌入结果在CPU上

        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)

        np.save(save_path_npy + prot_name + "_pt.npy", features[0])

        # 释放内存
        del input_ids, attention_mask, embedding, features
        gc.collect()  # 进行垃圾回收

print("ProtTrans embeddings generation completed ...")
