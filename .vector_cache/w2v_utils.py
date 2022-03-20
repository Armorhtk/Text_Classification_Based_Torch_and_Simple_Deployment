from gensim.models import KeyedVectors

# 词向量读取
# 词向量路径 腾讯词向量下载地址：https://ai.tencent.com/ailab/nlp/zh/download.html
Tencent_word2vec_path = r'./Tencent_AILab_ChineseEmbedding.txt'
# 保存前多少个词
limit_words = 50000
# 使用gensim读取词向量模型
model = KeyedVectors.load_word2vec_format(Tencent_word2vec_path,
                                          binary=False,
                                          limit=limit_words)
# 设置保存路径
save_word2vec_path = './{}-word2vec.txt'.format(limit_words)
# 保存词向量文件
model.save_word2vec_format(save_word2vec_path)
