import os
import json
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

FAISS_INDEX_PATH   = "/Users/priyansh/Downloads/faiss_index.index"
METADATA_JSON_PATH = "/Users/priyansh/Downloads/faiss_metadata.json"
EMBED_MODEL        = "all-MiniLM-L6-v2"

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error("Gemini API key not found! Add it in Streamlit secrets or as an environment variable.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

@st.cache_resource(show_spinner="Loading AI & data...")
def load_resources():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    return index, metadata, embedder, llm

index, metadata, embedder, llm = load_resources()

def search_faiss(query, top_k=4):
    query_emb = embedder.encode([query]).astype("float32")
    D, I = index.search(query_emb, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        rec = metadata[idx].copy()
        rec["score"] = float(dist)
        results.append(rec)
    return results

def build_rag_prompt(context_chunks, user_question):
    context_text = ""
    for c in context_chunks:
        context_text += (
            f"\n---\nTitle: {c.get('title','')}\n"
            f"URL: {c.get('video_url','')}\n"
            f"Excerpt: {c.get('chunk_text','')[:800]}...\n"
        )
    prompt = f"""You are an expert YouTube Growth Coach.
Using ONLY the information in the following video excerpts (with titles and URLs), answer the user's question as specifically and creatively as possible.

{context_text}

User's question:
{user_question}

Instructions:
- Reference video titles/URLs as sources when possible.
- If the user asks for a script or hook, write a detailed YouTube script/hook in the creator's style.
- Be clear, actionable, and step-by-step.
"""
    return prompt

st.set_page_config(page_title="YouTube Growth RAG Chatbot", page_icon="🎬")
st.title("🎬 YouTube Growth RAG Chatbot")
st.caption("Ask for channel tips, script ideas, viral hooks, and more — grounded in YOUR YouTube channel's videos.")

# User input
with st.form("chat_form"):
    user_question = st.text_input("Ask your YouTube channel question (growth, scripts, hooks, optimization, etc.):")
    top_k = st.slider("How many video sources to use?", 2, 8, 4)
    submitted = st.form_submit_button("Get AI Advice")

if submitted and user_question:
    with st.spinner("Thinking..."):
        chunks = search_faiss(user_question, top_k=top_k)
        prompt = build_rag_prompt(chunks, user_question)
        response = llm.invoke(prompt)

    st.markdown("### 🤖 AI's Answer")
    st.write(response.content)
    st.markdown("### 🔗 Source Videos & Chunks")
    for c in chunks:
        st.markdown(f"**{c.get('title','') or '(no title)'}** &nbsp; ([watch]({c.get('video_url','')})) &nbsp; — *distance: {c.get('score'):.4f}*")
        with st.expander("Show transcript excerpt"):
            st.write(c.get("chunk_text","")[:1500] + "...")

st.markdown("---")
st.caption("Built with FAISS, LangChain, Gemini, and Streamlit — 100% open & free.")

st.markdown(
    "<small>Tip: To keep your bot private, deploy with Streamlit Cloud's invite-only or HuggingFace Spaces private mode.</small>",
    unsafe_allow_html=True
)
