import os
# 强制设置 HF 镜像 (需在导入 sentence_transformers 前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
from rag.pipeline import RAGPipeline
from rag.config import Config

# 页面配置
st.set_page_config(page_title="DeepSeek中文医疗问答RAG", layout="wide")

st.title("🏥 基于RAG的中文医疗智能问答系统")
st.markdown("""
> **声明**：本系统仅用于课程演示与信息检索技术研究，生成内容不构成任何医疗建议。身体不适请务必前往正规医院就诊。
""")

# 初始化 Pipeline (利用st.cache_resource缓存，避免每次重载模型)
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

# Sidebar 配置
with st.sidebar:
    st.header("⚙️ 参数设置")
    top_k = st.slider("检索数量 (Top K)", 1, 10, Config.TOP_K_DEFAULT)
    min_sim = st.slider("相似度阈值 (Min Similarity)", 0.0, 1.0, Config.MIN_SIMILARITY_THRESHOLD)
    
    st.divider()
    st.info(f"📚 当前知识库大小: {pipeline.db.count()} chunks")
    if st.button("查看系统配置"):
        st.json({
            "DeepSeek Model": Config.DEEPSEEK_CHAT_MODEL,
            "Embedding": Config.DEEPSEEK_EMBEDDING_MODEL or Config.LOCAL_EMBEDDING_MODEL,
            "Chroma Path": Config.CHROMA_PATH
        })

# 主界面聊天框
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("查看引用证据 (Source Chunks)"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**证据 {idx+1}** (ID: {src['metadata'].get('sample_id')})")
                    st.text(src['text'])
                    st.divider()

# 输入框
question = st.chat_input("请输入医疗相关问题，例如：糖尿病有哪些典型症状？")

if question:
    # 1. 用户提问上屏
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 2. 调用RAG
    with st.chat_message("assistant"):
        with st.spinner("正在检索医疗文献并生成回答..."):
            result = pipeline.ask(question, top_k=top_k, min_similarity=min_sim)
        
        answer = result['answer']
        contexts = []
        # 组装 context 数据结构用于前端展示
        for i, doc in enumerate(result['context']):
            contexts.append({
                "text": doc,
                "metadata": result['metadatas'][i]
            })

        st.markdown(answer)
        
        # 3. 证据展示
        if contexts:
            with st.expander("查看引用证据 (Source Chunks)"):
                for idx, src in enumerate(contexts):
                    st.markdown(f"**证据 {idx+1}** (ID: {src['metadata'].get('sample_id')})")
                    st.text(src['text'])
                    st.divider()
        elif "无法确定" in answer:
            st.warning("未检索到有效证据，已触发拒答机制。")

    # 4. 保存历史
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": contexts
    })