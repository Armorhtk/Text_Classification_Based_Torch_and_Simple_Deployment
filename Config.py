# General parameters
SEED = 1234
max_length = 32
batch_size = 256
epochs = 500
learning_rate = 5e-4
early_stopping_nums = 10
best_metric = "f1" # 可选acc、p、r、f1

# model lists  Capsule Transformer TextRNN_Attention FastText
model_name = "TextCNN"
load_embedding = "None"
embedding_dim = 200

# embedding lists
w2v_file = "50000-word2vec.txt"
glove_file= "50000-word2vec.txt"

# data label list
label_list = ['news_story', 'news_culture', 'news_entertainment', 'news_sports',
              'news_finance', 'news_house', 'news_car', 'news_edu', 'news_tech',
              'news_military', 'news_travel', 'news_world', 'news_stock',
              'news_agriculture', 'news_game']
class_number = len(label_list)
if class_number == 2:
    type_avrage = 'binary'
else:
    """micro、macro"""
    type_avrage = 'macro'

# data list
task_name = "TNEWS"
data_path = 'dataset'
train_file = 'train.csv'
valid_file = 'dev.csv'
test_file = 'dev.csv'
predict_file = 'predict.txt'
result_file = 'result.csv'






