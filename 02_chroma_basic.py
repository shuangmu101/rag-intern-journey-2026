import chromadb
from sentence_transformers import SentenceTransformer


client = chromadb.PersistentClient(path="./chroma_db")
print("客户端已启动，数据将保存在 ./chroma_db 目录。\n")


collection = client.get_or_create_collection(name="tech_notes")


documents = [
    "反向传播算法通过计算损失函数对权重的梯度，利用链式法则逐层更新神经网络参数。",
    "Python 中的装饰器是一个接受函数作为参数并返回新函数的可调用对象，常用于日志、权限校验。",
    "RAG（检索增强生成）结合了信息检索和文本生成，先检索相关文档再交给大模型生成答案，可有效减少幻觉。",
    "Docker 容器通过 Namespace 实现进程隔离，通过 Cgroups 实现资源限制，比虚拟机更轻量。",
    "Transformer 模型的核心是自注意力机制，它允许模型在处理一个词时关注句子中的其他词，捕捉长距离依赖。",
]

# 每条文档需要唯一的 ID
ids = [f"doc_{i}" for i in range(len(documents))]
print("文本数据准备完毕，共", len(documents), "条。\n")


# 4. 生成向量（Embedding）—— 用 BGE 模型把文本变成高维向量
print("正在加载模型 (from cache)...")
model = SentenceTransformer('BAAI/bge-small-zh')  # 如果下载慢，可暂时换成 paraphrase-multilingual-MiniLM-L12-v2
embeddings = model.encode(documents).tolist()  # Chroma 需要 Python 列表格式
print("向量生成完成。\n")



# 5. 将文档、向量、ID 一起存入集合（add）

collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=ids
)


# 6. 用自然语言查询（query）—— 模拟用户提问

query_text = input("\n请输入你想查询的问题: ")
query_embedding = model.encode([query_text]).tolist()

# 检索 top-2 最相关的文档
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

# =============================================================================
# 7. 打印检索结果
# =============================================================================
print("\n" + "="*60)
print("检索结果（Top 2）")
print("="*60)
for i, (doc_id, distance, doc) in enumerate(
    zip(results['ids'][0], results['distances'][0], results['documents'][0])
):
    print(f"\n第 {i+1} 条：")
    print(f"  ID       : {doc_id}")
    print(f"  距离     : {distance:.4f}  (越小越相关)")
    print(f"  文档内容 : {doc}")

print("\n任务完成！")