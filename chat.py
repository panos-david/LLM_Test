#Run with:  streamlit run chat.py

import os
import pickle
import textwrap
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import torch
import re


# Configuration 

MODEL_PATH = "./pythia-main/models/pythia-410m"
EMBED_MODEL = "all-MiniLM-L6-v2" 
CHUNK_SIZE = 800 
INDEX_PATH = Path("faiss.index")
DOCS_PATH = Path("docs.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Lazy loaders

@st.cache_resource(show_spinner="Loading Pythia…")
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto" if DEVICE == "cuda" else None,
    )
    return tokenizer, model


@st.cache_resource(show_spinner="Loading embedder…")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL, device=DEVICE)



# Knowledge‑base helpers


def chunk_text(text: str, size: int = CHUNK_SIZE):
    """Simple character‑based splitter; swap out for tokenizer‑aware splitter if needed."""
    return textwrap.wrap(text, size)


def ensure_index(dim: int):
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatL2(dim)
    return index


def add_documents(uploaded_files):
    """Read uploaded txt files ➜ chunk ➜ embed ➜ add to FAISS."""
    embedder = load_embedder()
    new_chunks = []
    for f in uploaded_files:
        content = f.read().decode("utf-8", errors="ignore")
        new_chunks.extend(chunk_text(content))

    vectors = embedder.encode(new_chunks, batch_size=32, show_progress_bar=True)
    index = ensure_index(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    if DOCS_PATH.exists():
        existing = pickle.loads(DOCS_PATH.read_bytes())
        all_chunks = existing + new_chunks
    else:
        all_chunks = new_chunks
    DOCS_PATH.write_bytes(pickle.dumps(all_chunks))

    return len(new_chunks)


def retrieve(question: str, k: int = 4):
    if not (INDEX_PATH.exists() and DOCS_PATH.exists()):
        return []

    embedder = load_embedder()
    q_vec = embedder.encode([question])
    index = faiss.read_index(str(INDEX_PATH))
    D, I = index.search(q_vec, k)
    chunks = pickle.loads(DOCS_PATH.read_bytes())
    return [chunks[i] for i in I[0]]


def generate_answer(question: str):
    tokenizer, model = load_llm()
    contexts = retrieve(question)
    if not contexts:
        return "❗ Upload some documents first, then ask a question."


    prompt = (
        "You are an assistant. Use the following background information to answer "
        "the question **without** repeating or quoting it directly.\n\n"
    )
    for i, c in enumerate(contexts, 1):
        prompt += f"### Context {i}\n{c}\n"
    prompt += f"\n### Question\n{question}\n### Answer\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = text.split("### Answer")[-1]

    answer = "\n".join(
        line for line in answer.splitlines()
        if not line.lower().startswith(("context", "### context"))
    ).strip()

    return answer



# UI


st.title("AI App")

st.markdown(
    "Create your private Knowledge Base!")

uploaded = st.file_uploader("Add document(s)", type=["txt"], accept_multiple_files=True)
if uploaded and st.button("↪️ Add to your Knowledge Base"):
    with st.spinner("Processing"):
        n_chunks = add_documents(uploaded)
    st.success("Success!")

q = st.text_input("Ask your question!")
if q:
    with st.spinner("Answering..."):
        ans = generate_answer(q)
    st.markdown("Here's what I found:")
    st.write(ans)

    with st.expander("Resources"):
        for c in retrieve(q):
            st.markdown("---")
            st.write(c)

