# DeepSS2GO_v2_colab 使用文档


写在前面：
本算法一共分为2个part，
PART1，利用SPOT-1D-LM将一级序列aa转化为二级结构ss8
PART2, 结合aa + ss8 + Diamond进行功能预测

Colab有免费版和付费版。
Free版的T4 GPU：显存15.0G，RAM为12.7G
Pro版的 L4 GPU：显存22G，RAM为53G

本算法PART1中用到了模型ProtTrans，免费版的GPU显存不够，CPU RAM不够。

- 如果是Colab付费用户，建议使用`DeepSS2GO_v2_colab_pro.ipynb`，只需要在第一步上传fasta文件，即可一键运行run all。

- 如果是Colab免费用户，建议使用`DeepSS2GO_v2_colab_free.ipynb`，需要自己先在其他网站预测出对应氨基酸一级序列的二级结构，然后将两者一并上传。几个二级结构预测网站，仅供参考：

```bash
0. psipred: 
http://bioinf.cs.ucl.ac.uk/psipred/

1. JPred: 
http://www.compbio.dundee.ac.uk/jpred/index.html

2. ProtPredicct: 
http://predictprotein.org
```


## PART0


- `DeepSS2GO_v2_colab_clean.ipynb`为clean版本
- `DeepSS2GO_v2_colab_example.ipynb`为包含运行数据的版本

## 使用方法

1. 在colab中加载`DeepSS2GO_v2_colab_clean.ipynb`

2. Runtime - run all

3. 上传文件，


![Img](./FILES/README_v2_colab.md/img-20240606152119.png)


## tips: 

因为加载库的先后顺序，会弹出，选择cancel即可。

![Img](./FILES/README_v2_colab.md/img-20240606152358.png)



