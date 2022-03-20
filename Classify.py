import jieba
import torch
import DataSet
import pandas as pd
import os
import re
from Config import max_length, label_list,model_name,data_path,task_name,predict_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def x_tokenize(x):
    str = re.sub('[^\u4e00-\u9fa5]', "", x)
    return jieba.lcut(str)

def getModel(name):
    model = torch.load('done_model/'+name+'_model.pkl')
    return model

def model_predict(model, sentence):
    model.eval()
    tokenized = x_tokenize(sentence)
    indexed = [DataSet.getTEXT().vocab.stoi[t] for t in tokenized]
    if(len(indexed) > max_length):
        indexed = indexed[:max_length]
    else:
        for i in range(max_length-len(indexed)):
            indexed.append(DataSet.getTEXT().vocab.stoi['<pad>'])
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    tensor = torch.t(tensor)
    out = torch.softmax(model(tensor),dim=0)
    result = label_list[out.argmax(0).cpu().numpy()]
    response = {k:v for k,v in zip(label_list,out.cpu().detach().numpy())}
    return response,result

def predict_csv(model):
    model.eval()
    outs = []
    df = pd.read_csv(os.path.join(data_path,task_name,predict_file),sep='\t')
    for sentence in df["query"]:
        tokenized = x_tokenize(sentence)
        indexed = [DataSet.getTEXT().vocab.stoi[t] for t in tokenized]
        if(len(indexed) > max_length):
            indexed = indexed[:max_length]
        else:
            for i in range(max_length-len(indexed)):
                indexed.append(DataSet.getTEXT().vocab.stoi['<pad>'])
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        tensor = torch.t(tensor)
        out = model(tensor)
        result = label_list[out.argmax(0).cpu().numpy()]
        outs.append(result)
    df["label"] = outs
    sumbit = df[["query","label"]]
    sumbit.to_csv(os.path.join(data_path,task_name,"predict.txt"),index=False,sep='\t')

def load_model():
    model = getModel(model_name)
    model = model.to(device)
    return model

if __name__=="__main__":
    model = load_model()
    sent1 = '你要那么确定百分百，家里房子卖了，上'
    response, result = model_predict(model, sent1)
    print('概率分布：{}\n预测标签：{}'.format(response,result))
    # predict_csv(model)
