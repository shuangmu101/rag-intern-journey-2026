import os
import requests
from sentence_transformers import SentenceTransformer
import chromadb

# ==================== 配置区 ====================
API_KEY = "sk-435345da9e994bd98f68c496aa0b9a08"   # 替换成你的
API_URL = "https://api.deepseek.com/v1/chat/completions"  # 以DeepSeek为例
MODEL_NAME = "deepseek-chat"            # 或 glm-4-flash 等
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh"

# 硬编码的长文本（模拟你的“知识库”，可以用昨天的多条文本）
LONG_TEXT = """
反向传播算法是训练人工神经网络的核心算法。它通过计算损失函数对每个权重的梯度，并利用梯度下降法更新权重。
Python装饰器是一种语法糖，允许在不修改原函数代码的情况下增加额外功能，常用于日志、权限验证等场景。
RAG（检索增强生成）是一种将检索与生成结合的框架，首先从外部知识库检索相关文档片段，然后将这些片段与问题一起交给大语言模型生成答案。
自注意力机制是Transformer模型的基础，它让模型在处理每个单词时能够关注句子中所有其他单词，从而捕捉上下文信息。
Docker是一种容器化技术，它将应用及其依赖打包在一个轻量级、可移植的容器中，实现环境一致性。通过Dockerfile构建镜像，通过镜像运行容器。
"""

# ==================== 1. 切片函数（简易版，按句子边界切） ====================
def simple_split(text, max_chars=300, overlap=50):
    """简易切片：按句子切，每个块不超过 max_chars，有 overlap"""
    sentences = text.replace('\n', ' ').split('。')
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) <= max_chars:
            current_chunk += sent + "。"
        else:
            chunks.append(current_chunk.strip())
            # 重叠部分取 current_chunk 的最后 overlap 个字符
            current_chunk = current_chunk[-overlap:] + sent + "。"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ==================== 2. 构建向量库 ====================
def build_index():
    print("正在切片并创建向量库...")
    chunks = simple_split(LONG_TEXT, max_chars=300, overlap=50)
    print(f"共得到 {len(chunks)} 个文本块。")

    # 初始化 Chroma（内存模式，今日实验够用）
    client = chromadb.Client()
    collection = client.get_or_create_collection("minimal_rag")

    # 嵌入模型
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(chunks).tolist()

    # 存入（如果集合已存在先清空，每次运行重新构建）
    if collection.count() > 0:
        collection.delete(ids=collection.get()['ids'])
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    return collection, model

# ==================== 3. 检索 ====================
def retrieve(query, collection, model, top_k=3):
    query_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    return results['documents'][0]  # 返回文本列表

# ==================== 4. 调用大模型 ====================
def ask_llm(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个乐于助人的助手，基于给定的资料回答问题。如果资料不足以回答，就如实说不知道。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"API调用失败: {response.text}"

# ==================== 5. 主流程：构建RAG ====================
if __name__ == "__main__":
    # 初始化向量库（只需要执行一次，也可放在外面单独预处理）
    collection, embed_model = build_index()

    print("\n=== 简单RAG问答已准备就绪 ===")
    while True:
        query = input("\n请输入你的问题（输入 quit 退出）: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break

        # 检索
        retrieved_chunks = retrieve(query, collection, embed_model, top_k=3)
        # 构造 prompt
        context = "\n\n---\n\n".join(retrieved_chunks)
        prompt = f"""请根据以下参考资料回答问题。

参考资料：
{context}

问题：{query}
答案："""

        print("\n正在调用大模型生成答案...")
        answer = ask_llm(prompt)
        print(f"\n回答：{answer}")
        # 可选：打印检索到的资料（方便调试）
        print("\n[检索到的资料摘要]")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"  {i}. {chunk[:80]}...")