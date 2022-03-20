import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
from Config import *
import DataSet
import warnings
warnings.filterwarnings("ignore")
from model.TextCNN import TextCNN
from model.TextRCNN import TextRCNN
from model.TextRNN import TextRNN
from model.TextRNN_Attention import TextRNN_Attention
from model.Transformer import Transformer
from model.FastText import FastText
from model.DPCNN import DPCNN
from model.Capsule import Capsule

model_select = {"TextCNN":TextCNN(),
                "TextRNN":TextRNN(),
                "TextRCNN":TextRCNN(),
                "TextRNN_Attention":TextRNN_Attention(),
                "Transformer":Transformer(),
                "FastText":FastText(),
                "DPCNN":DPCNN(),
                "Capsule":Capsule(),
                }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mutil_metrics(y_true,y_predict,type_avrage='macro'):
    result = {}
    result["acc"] = accuracy_score(y_true=y_true,y_pred=y_predict)
    result["precision"] = precision_score(y_true=y_true,y_pred=y_predict,average=type_avrage)
    result["recall"] = recall_score(y_true=y_true, y_pred=y_predict, average=type_avrage)
    result["f1"] = f1_score(y_true=y_true, y_pred=y_predict, average=type_avrage)
    # result["kappa"] = cohen_kappa_score(y1=y_true, y2=y_predict)
    for k,v in result.items():
        result[k] = round(v,5)
    return result

def test_model(test_iter, name, device):
    model = torch.load('done_model/'+name+'_model.pkl')
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    for batch in test_iter:
        feature = batch.text
        target = batch.label
        with torch.no_grad():
            feature = torch.t(feature)
        feature, target = feature.to(device), target.to(device)
        out = model(feature)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
    result = mutil_metrics(y_true, y_pred, type_avrage='macro')
    print('>>> Test {} Result:{} \n'.format(name,result))
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=label_list, digits=3))
    save_experimental_details(result)

def train_model(train_iter, dev_iter, model, name, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.6)
    model.train()
    best_score = 0
    early_stopping = 0
    print('training...')
    # 每一个epoch
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_score = 0
        total_train_num = len(train_iter)
        progress_bar = tqdm(enumerate(train_iter), total=len(train_iter))
        # 每一步等于一个Batch
        for i,batch in progress_bar:
            feature = batch.text
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            # 输入模型,得到输出概率分布
            logit = model(feature)
            # 使用损失函得到损失
            loss = F.cross_entropy(logit, target)
            # 反向传播梯度
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            # 计算评估分数
            result = mutil_metrics(target.cpu().numpy(),torch.argmax(logit, dim=1).cpu().numpy(),type_avrage='macro')
            train_score += result[best_metric]
        # 进入验证阶段
        print('>>> Epoch_{}, Train loss is {}, {}:{} \n'.format(epoch,loss.item()/total_train_num, best_metric,train_score/total_train_num))
        model.eval()
        total_loss = 0.0
        valid_true = []
        valid_pred = []
        total_valid_num = len(dev_iter)
        for i, batch in enumerate(dev_iter):
            feature = batch.text  # (W,N) (N)
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            valid_true.extend(target.cpu().numpy())
            valid_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
        valid_result = mutil_metrics(valid_true, valid_pred, type_avrage='macro')
        print('>>> Epoch_{}, Valid loss:{}, {}:{} \n'.format(epoch, total_loss/total_valid_num, best_metric,valid_result[best_metric]))
        if(valid_result[best_metric] > best_score):
            early_stopping = 0
            print('save model...')
            best_score = valid_result[best_metric]
            saveModel(model, name=name)
        else:
            early_stopping += 1
        if early_stopping == early_stopping_nums:
            break

def saveModel(model,name):
    torch.save(model, 'done_model/'+name+'_model.pkl')

def save_experimental_details(test_result):
    save_path = os.path.join(data_path,result_file)
    var = [task_name, model_name,load_embedding,w2v_file, SEED, batch_size, max_length, learning_rate]
    names = ['task_dataset','model_name',"embedding_type","w2v_name",'Seed','Batch_size','Max_lenth','lr']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_result, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    if not os.path.exists(save_path):
        ori = []
        ori.append(values)
        new_df = pd.DataFrame(ori, columns=keys)
        new_df.to_csv(os.path.join(data_path,result_file), index=False)
    else:
        df = pd.read_csv(save_path)
        new = pd.DataFrame(results, index=[1])
        df = df.append(new, ignore_index=True)
        df.to_csv(save_path, index=False)
    data_diagram = pd.read_csv(save_path)
    print('test_results \n', data_diagram)


if __name__ == '__main__':
    model = model_select[model_name]
    train_iter, val_iter, test_iter = DataSet.getIter()
    train_model(train_iter, val_iter, model, model_name, device)
    test_model(test_iter, model_name, device)