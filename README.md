
Text classification practice code and simple deployment based on Torch and deep neural network

基于torch和深度神经网络的文本分类实践代码和简单部署

*********************** **更新** ***********************
- 2022/03/20 开源基于深度神经网络的文本分类训练和部署代码(包含TextCNN、TextRNN、DPCNN等模型)

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
待更新

# 实验结果
待实验

# 依赖版本
- torch-1.8.0
- torchtext-0.9.0
- flask-2.0.2

# 鸣谢
- [NTDXYG/Text-Classify-based-pytorch](https://github.com/NTDXYG)
- [shawroad/Text-Classification-Pytorch](https://github.com/shawroad/Text-Classification-Pytorch)

