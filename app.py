"""ChannelMind — RAG chatbot over a YouTube channel's transcripts.

Retrieval: MiniLM embeddings + FAISS (exact L2) with a relevance threshold.
Generation: Gemini via langchain-google-genai. Index artifacts live in
data/ and are produced by scripts/build_index.py.
"""

import os

import streamlit as st

import config
from core.retrieval import (
    IndexMetadataMismatch,
    build_bm25,
    load_cross_encoder,
    load_index_pair,
    retrieve,
)

st.set_page_config(page_title="ChannelMind — YouTube RAG", page_icon="🎬")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error(
        "Gemini API key not found. Add `GOOGLE_API_KEY` in Streamlit secrets "
        "(HF Spaces: Settings → Variables and secrets) or as an environment variable."
    )
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


@st.cache_resource(show_spinner="Loading index and models...")
def load_resources():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from sentence_transformers import SentenceTransformer

    index, metadata = load_index_pair(config.INDEX_PATH, config.METADATA_PATH)
    embedder = SentenceTransformer(config.EMBED_MODEL)
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        temperature=config.GEN_TEMPERATURE,
        max_output_tokens=config.GEN_MAX_OUTPUT_TOKENS,
    )
    # BM25 is cheap to build in-memory; the cross-encoder is only loaded when
    # reranking is enabled so a dense/hybrid deploy never pays its cold start.
    bm25 = build_bm25(metadata["chunks"]) if config.HYBRID else None
    cross_encoder = load_cross_encoder(config.CROSS_ENCODER_MODEL) if config.RERANK else None
    return index, metadata, embedder, llm, bm25, cross_encoder


try:
    index, metadata, embedder, llm, bm25, cross_encoder = load_resources()
except FileNotFoundError as exc:
    st.error(
        f"{exc}\n\nThe FAISS index and metadata pair is missing. From the repo root run:\n\n"
        "```\npython scripts/build_index.py\n```"
    )
    st.stop()
except IndexMetadataMismatch as exc:
    st.error(f"Index artifacts are inconsistent: {exc}")
    st.stop()

chunks = metadata["chunks"]
build_info = metadata.get("build_info", {})
channel_name = chunks[0].get("channel_name", "this channel") if chunks else "this channel"


def build_rag_prompt(context_chunks, user_question):
    context_text = ""
    for c in context_chunks:
        context_text += (
            f"\n---\nTitle: {c.get('title', '')}\n"
            f"URL: {c.get('video_url', '')}\n"
            f"Excerpt: {c.get('chunk_text', '')[:config.EXCERPT_CHARS]}\n"
        )
    return f"""You are an expert YouTube growth coach answering questions about the channel "{channel_name}".
Use ONLY the information in the following video excerpts (with titles and URLs).

{context_text}

User's question:
{user_question}

Instructions:
- Cite video titles/URLs as sources whenever possible.
- If the user asks for a script or hook, write it in the creator's style.
- Be clear, actionable, and step-by-step.
- If the excerpts do not contain the answer, say so instead of inventing one.
"""


st.title("🎬 ChannelMind")
st.caption(
    f"RAG chatbot grounded in **{channel_name}**'s videos — "
    f"{build_info.get('num_videos_with_transcript', '?')} transcribed videos, "
    f"{build_info.get('num_chunks', len(chunks))} searchable chunks."
)

with st.sidebar:
    st.subheader("Corpus")
    st.write(f"Channel: `{build_info.get('channel_id', config.CHANNEL_ID)}`")
    st.write(f"Embeddings: `{build_info.get('embed_model', config.EMBED_MODEL)}`")
    st.write(f"LLM: `{config.GEMINI_MODEL}`")
    _mode = "dense"
    if config.HYBRID:
        _mode = "hybrid + rerank" if config.RERANK else "hybrid (BM25+dense)"
    elif config.RERANK:
        _mode = "dense + rerank"
    st.write(f"Retrieval: `{_mode}`")
    st.write(f"Built: {build_info.get('built_at', 'unknown')}")

with st.form("chat_form"):
    user_question = st.text_input("Ask about growth, scripts, hooks, titles, thumbnails...")
    top_k = st.slider("How many chunks to retrieve?", 2, 8, config.TOP_K)
    submitted = st.form_submit_button("Ask")

if submitted and user_question:
    with st.spinner("Retrieving..."):
        results = retrieve(
            index,
            chunks,
            embedder,
            user_question,
            top_k=top_k,
            hybrid=config.HYBRID,
            rerank=config.RERANK,
            bm25=bm25,
            cross_encoder=cross_encoder,
            candidate_k=config.CANDIDATE_K,
            rrf_k=config.RRF_K,
            max_distance=config.DISTANCE_THRESHOLD,
        )

    if not results:
        st.warning(
            "No relevant content found in this channel's videos for that question. "
            "Try rephrasing it around YouTube growth, titles, thumbnails, or scripting."
        )
    else:
        try:
            with st.spinner("Generating answer..."):
                response = llm.invoke(build_rag_prompt(results, user_question))
        except Exception as exc:  # noqa: BLE001 — surface any Gemini-side failure cleanly
            st.error(
                "The answer couldn't be generated just now. This is usually a "
                "temporary Gemini API issue (rate limit, quota, network, or a "
                "content safety block). Please try again in a moment.\n\n"
                f"Details: `{type(exc).__name__}: {exc}`"
            )
            st.stop()

        st.markdown("### Answer")
        st.write(response.text)  # .content is a list of blocks in langchain-core >= 1.0

        st.markdown("### Sources")
        seen = set()
        for c in results:
            key = c.get("video_id")
            title = c.get("title") or "(untitled)"
            url = c.get("video_url", "")
            if key not in seen:
                st.markdown(f"- [{title}]({url}) — distance {c['score']:.3f}")
                seen.add(key)
        with st.expander("Show retrieved excerpts"):
            for c in results:
                st.markdown(f"**{c.get('title', '')}** (chunk {c.get('chunk_index')})")
                st.write(c.get("chunk_text", "")[:1500])

st.markdown("---")
st.caption("Local MiniLM embeddings + FAISS retrieval, Gemini generation. Built with Streamlit.")
