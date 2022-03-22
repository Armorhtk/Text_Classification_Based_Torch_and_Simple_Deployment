import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def TongHuaShun_task():
    curr_path = "./TongHuaShun"
    df = pd.read_csv(os.path.join(curr_path,"orgin.txt"),sep='\t')
    df['len'] = [len(i) for i in df["query"]]
    print("上四分位点:{}".format(df.quantile(.75)))
    print("90%的文本长度小于{}".format(df.quantile(.9)))
    df["len"].hist(bins=20)
    plt.show()
    df = df[["query","label"]]
    df.columns = ["text","label"]
    df["label"] = [1 if i=="攻击用户" else 0 for i in df["label"]]
    train,test,_,_ = train_test_split(df,df["label"],test_size=0.1,random_state=42)
    train.to_csv(os.path.join(curr_path,"train.csv"),index=False)
    test.to_csv(os.path.join(curr_path,"test.csv"),index=False)
    print(train.label.value_counts())
    print(test.label.value_counts())

def TNEWS_task(file_name,save_name):
    curr_path = "./TNEWS"
    label = pd.read_json(os.path.join(curr_path,"labels.json"),lines=True)
    print(label["label_desc"].unique())
    label_dict = {l: idx for idx, l in enumerate(label["label_desc"].unique())}
    with open(os.path.join(curr_path,"label_dict.json"),"w",encoding="utf-8") as f:
        json.dump(label_dict,f)
    df = pd.read_json(os.path.join(curr_path,file_name),orient="values",lines=True)
    df["label"] = [label_dict[i] for i in df["label_desc"]]
    df = df[["sentence","label"]]
    df.columns = ["text", "label"]
    df.to_csv(os.path.join(curr_path,save_name),index=False)


if __name__ == '__main__':
    TNEWS_task("train.json","train.csv")
    TNEWS_task("dev.json", "dev.csv")
