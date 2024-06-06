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

# 创建保存路径
save_path_npy = path_base + 'redundancy/SPOT1DLM_inputs_new/'
os.makedirs(save_path_npy, exist_ok=True)  # 使用os.makedirs创建目录，并确保目录存在

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', type=str, help='define the device you want the')
args = parser.parse_args()


# ### original:
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

# # 加载模型和分词器
# tokenizer = T5Tokenizer.from_pretrained(path_Prot_T5_XL_UniRef50, do_lower_case=False)
# model = T5EncoderModel.from_pretrained(path_Prot_T5_XL_UniRef50)

# 设置设备
device = torch.device(args.device)
model = model.to(device)
model = model.eval()

# 读取数据
prot_df = pd.read_pickle(path_base + 'pub_data/data_new/new_clean_aa.pkl')

# 处理数据
k = 0
for index, row in prot_df.iterrows():
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

    if device.type == "cpu":
        embedding = embedding.last_hidden_state.numpy()
    else:
        embedding = embedding.last_hidden_state.cpu().numpy()  # 将embedding移动到CPU以减少GPU显存占用

    # 提取特征
    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    # 保存特征
    np.save(os.path.join(save_path_npy, prot_name + "_pt.npy"), features[0])

    # 清理显存
    del input_ids, attention_mask, embedding, features  # 删除不再使用的变量以释放内存
    torch.cuda.empty_cache()  # 清空缓存以减少显存占用

print("ProtTrans embeddings generation completed ...")
