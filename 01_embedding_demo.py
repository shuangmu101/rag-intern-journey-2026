from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载中文嵌入模型（如果下载慢，可先换成 paraphrase-multilingual-MiniLM-L12-v2）
model = SentenceTransformer('BAAI/bge-small-zh')

# 2. 定义三句话，观察语义距离
sentences = [
    "今天天气真不错",      # A
    "外面阳光明媚",        # B (和A近)
    "反向传播算法很难理解"  # C (和A远)
]

# 3. 生成向量
embeddings = model.encode(sentences)

# 4. 计算并打印余弦相似度
print("【语义相似度实验】")
print(f"A vs B (天气相关): {cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]:.4f}")
print(f"A vs C (天气 vs 算法): {cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]:.4f}")

# 5. 直观感受：两个数字应该有明显差距（比如 0.85 vs 0.32）
