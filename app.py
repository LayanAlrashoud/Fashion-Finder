from dotenv import load_dotenv
import json, base64, re
from typing import Optional, List, Dict, Any

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


st.set_page_config(page_title="Fashion Finder", layout="wide")

load_dotenv(override=True)
client = OpenAI()


# ================= LOAD VECTOR DB =================

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)


vectorstore = load_vectorstore()


# ================= IMAGE UNDERSTANDING =================

def describe_image_bytes(img_bytes: bytes, mime: str) -> dict:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Describe this product image for e-commerce search. "
                        "Return STRICT JSON only with keys: "
                        "caption, attributes(type, gender, color, material, style)."
                    )
                },
                {"type": "input_image", "image_url": data_url}
            ],
        }],
    )

    text = resp.output_text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except:
        return {"caption": text, "attributes": {}}


# ================= DATA MODELS =================

class QueryConstraints(BaseModel):
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    gender: Optional[str] = None
    color: Optional[str] = None
    product_type: Optional[str] = None
    must_have_keywords: List[str] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)


class RerankResult(BaseModel):
    ranking: List[int] = Field(default_factory=list)
    top_reason: str = ""


# ================= NORMALIZATION =================

def normalize_text(s):
    return str(s).lower().strip() if s else None


def normalize_gender(g):
    if not g:
        return None
    g = g.lower()
    if g in ["women", "female", "lady"]:
        return "women"
    if g in ["men", "male"]:
        return "men"
    return None


# ================= LLM SETUP =================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
constraints_llm = llm.with_structured_output(QueryConstraints)
rerank_llm = llm.with_structured_output(RerankResult)


# ================= CONSTRAINT EXTRACTION =================

def extract_constraints(question):
    prompt = f"Extract shopping constraints from: {question}"
    c = constraints_llm.invoke(prompt)
    c.gender = normalize_gender(c.gender)
    return c


def merge_constraints(text_c, img_attrs):
    if not text_c.color:
        text_c.color = normalize_text(img_attrs.get("color"))
    if not text_c.product_type:
        text_c.product_type = normalize_text(img_attrs.get("type"))
    if not text_c.gender:
        text_c.gender = normalize_gender(img_attrs.get("gender"))
    return text_c


# ================= FILTER =================

def passes_constraints(doc, c):
    m = doc.metadata
    title = str(m.get("title", "")).lower()
    cat = str(m.get("categoryName", "")).lower()

    price = m.get("price")
    try:
        price = float(price) if price else None
    except:
        price = None

    if c.max_price is not None and price and price > c.max_price:
        return False
    if c.min_price is not None and price and price < c.min_price:
        return False

    if c.gender and c.gender not in cat:
        return False

    if c.product_type and c.product_type not in title:
        return False

    if c.color and c.color not in title:
        return False

    for kw in c.must_have_keywords:
        if kw not in title:
            return False

    for kw in c.exclude_keywords:
        if kw in title:
            return False

    return True


# ================= RETRIEVAL =================

def retrieve_docs(retriever, query):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


def build_query(text, caption, attrs):
    attrs_text = " ".join([f"{k}:{v}" for k, v in attrs.items() if v])
    return f"{text} {text} {caption} {attrs_text}"


# ================= RERANK =================

def rerank(question, caption, attrs, c, candidates):
    items = []
    for i, d in enumerate(candidates):
        items.append({
            "id": i,
            "title": d.metadata.get("title"),
            "price": d.metadata.get("price"),
            "category": d.metadata.get("categoryName"),
        })

    prompt = f"""
Rank products by relevance.

Query: {question}
Caption: {caption}
Attrs: {attrs}
Constraints: {c.model_dump()}

Return JSON ranking list.
{items}
"""

    rr = rerank_llm.invoke(prompt)
    ranking = rr.ranking if rr.ranking else list(range(len(candidates)))
    return [candidates[i] for i in ranking]


# ================= UI =================

st.title("Fashion Finder")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload Image")
    question = st.text_input("Describe what you want")
    run = st.button("Search")


# ================= SEARCH =================

if run:

    caption = ""
    attrs = {}

    if uploaded:
        img_bytes = uploaded.read()
        img_info = describe_image_bytes(img_bytes, uploaded.type)
        caption = img_info.get("caption", "")
        attrs = img_info.get("attributes", {})

    if question:
        constraints = extract_constraints(question)
    else:
        constraints = QueryConstraints()

    constraints = merge_constraints(constraints, attrs)

    query = build_query(question, caption, attrs)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 30, "fetch_k": 80}
    )

    hits = retrieve_docs(retriever, query)

    filtered = [h for h in hits if passes_constraints(h, constraints)]
    hits = filtered or hits

    reranked = rerank(question, caption, attrs, constraints, hits[:15])
    results = reranked[:6]

    st.subheader("Results")

    for d in results:
        m = d.metadata
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.image(m.get("imgUrl"))
            with c2:
                st.markdown(f"**{m.get('title')}**")
                st.write(f"£ {m.get('price')}")
                st.markdown(f"[Open]({m.get('productURL')})")
