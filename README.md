
Text classification practice code and simple deployment based on Torch and deep neural network

基于torch和深度神经网络的文本分类实践代码和简单部署

*********************** **更新** ***********************
- 2022/03/20 开源基于深度神经网络的文本分类训练和部署代码(包含TextCNN、TextRNN、DPCNN等模型)
- 2020/03/22 更新在短文本数据集TNEWS上实验结果

# 仓库内容
- [仓库介绍](#仓库介绍)
- [运行流程](#运行流程)
- [数据说明](#数据说明)
- [实验结果](#实验结果)
- [依赖版本](#依赖版本)
- [鸣谢](#鸣谢)

# 仓库介绍
脚本参数解释
- Config.py 配置文件，存放数据和模型的超参数、各类文件路径配置等
- DataSet.py 数据集处理文件，使用torchtext统一处理为<text,label>形式
- Train_eval.py 训练和验证模型文件，训练的主入口
- Classify.py 预测文件，用于预测单条数据或文件数据
- app.py 部署文件，使用flask搭建的建议web应用
- model文件夹 存放各类模型
- done_model 存放训练后的最佳模型
- dataset 存放各类数据集，各数据集以任务名称命名，训练集、验证集、测试集文件名称和Config中配置的名称一致；
  内有data_process.py数据处理文件，可在此处写新任务的数据处理代码；result.csv记录了每次训练完后测试集的评价指标和本次训练的实验细节
- templates 存放web程序，html、css、js、images等，
- .vector_cache 存放词向量缓存文件，内有w2v_utils.py和使用说明，用于处理腾讯AI实验室的词向量文件

# 运行流程
- 训练：在Config中配置相关参数，进入Train_eval.py开始训练
- 部署：启动app.py，登录http://127.0.0.1:5000/ 进行在web页面进行文本分类交互 

# 数据说明
实验数据来自[中文语言理解测评基准(CLUE)平台](https://github.com/CLUEbenchmark) 的短文本数据集[TNEWS](https://cluebenchmarks.com/introduce.html)

下载地址：https://github.com/CLUEbenchmark/CLUE 

每一条数据有三个属性，从前往后分别是 分类ID，分类名称，新闻字符串（仅含标题）。

数据量：训练集(266,000)，验证集(57,000)，测试集(57,000)

例子：
```json
{ "label": "102",
  "label_des": "news_entertainment",
  "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物" }
```

# 实验结果

其中，Model列表内以“-None”为后缀表示不使用词向量，“-w2v”为后缀表示使用word2vec词向量
使用的词向量为腾讯开源的200维度词向量的前50000个词

|          Model         | Accuracy | Precision | Recall |   F1  |
|:----------------------:|:--------:|:---------:|:------:|:-----:|
|      FastText-None     |   0.439  |   0.449   |  0.412 | 0.414 |
|      FastText-w2v      |   0.555  |   0.548   |  0.519 | 0.523 |
|      TextCNN-None      |   0.478  |   0.495   |  0.450 | 0.461 |
|       TextCNN-w2v      |**0.562** | **0.559** |**0.537**|**0.544**|
|      TextRNN-None      |   0.386  |   0.376   |  0.351 | 0.355 |
|       TextRNN-w2v      |   0.502  |   0.488   |  0.477 | 0.481 |
|      TextRCNN-None     |   0.455  |   0.427   |  0.420 | 0.420 |
|      TextRCNN-w2v      |   0.546  |   0.547   |  0.506 | 0.515 |
| TextRNN_Attention-None |   0.444  |   0.438   |  0.409 | 0.418 |
|  TextRNN_Attention-w2v |   0.538  |   0.502   |  0.499 | 0.500 |
|       DPCNN-None       |   0.389  |   0.373   |  0.359 | 0.359 |
|        DPCNN-w2v       |   0.536  |   0.527   |  0.518 | 0.518 |
|    Transformer-None    |   0.485  |   0.478   |  0.453 | 0.460 |
|     Transformer-w2v    |   0.529  |   0.519   |  0.503 | 0.507 |
|      Capsule-None      |   0.483  |   0.484   |  0.454 | 0.465 |
|       Capsule-w2v      |   0.551  |   0.521   |  0.509 | 0.513 |

上表为验证集效果，没有上传到CLUE查看测试集结果。

从结果上看，TextCNN-w2v在实验数据集上表现最佳,次优模型是 FastText-w2v、Capsule-w2v。
若TextCNN-w2v能在测试集能取得同样效果（acc：0.562），则非常接近官方给出的BERT-Base的评测效果，说明腾讯开源的词向量质量很高。



# 依赖版本
- torch-1.8.0
- torchtext-0.9.0
- flask-2.0.2

# 鸣谢
- [CLUEbenchmark](https://github.com/CLUEbenchmark)
- [NTDXYG/Text-Classify-based-pytorch](https://github.com/NTDXYG)
- [shawroad/Text-Classification-Pytorch](https://github.com/shawroad/Text-Classification-Pytorch)


