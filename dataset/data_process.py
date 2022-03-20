import os
import pandas as pd
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

if __name__ == '__main__':
    TongHuaShun_task()
