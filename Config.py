# General parameters
SEED = 1234
max_length = 300
batch_size = 128
epochs = 500
learning_rate = 5e-4
early_stopping_nums = 10
best_metric = "f1" # 可选acc、p、r、f1

# model lists
model_name = "Capsule"
load_embedding = "w2v"
embedding_dim = 200

# embedding lists
w2v_file = "50000-word2vec.txt"
glove_file= "50000-word2vec.txt"

# data label list
label_list = ['未攻击用户', '攻击用户']
class_number = len(label_list)
if class_number == 2:
    type_avrage = 'binary'
else:
    """micro、macro"""
    type_avrage = 'macro'

# data list
task_name = "TongHuaShun"
data_path = 'dataset'
train_file = 'train.csv'
valid_file = 'test.csv'
test_file = 'test.csv'
predict_file = 'test_data_A.txt'
result_file = 'result.csv'






