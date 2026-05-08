import streamlit as st
import requests
import os

st.set_page_config(layout="wide", page_title="ML Paper RAG", page_icon="📄")

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

def api_get(endpoint, params=None):
    try:
        res = requests.get(f"{BASE_URL}{endpoint}", params=params)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return None

def api_post(endpoint, json=None, files=None):
    try:
        res = requests.post(f"{BASE_URL}{endpoint}", json=json, files=files)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return None

@st.cache_data(ttl=300)
def fetch_papers():
    papers = api_get("/papers")
    return papers if papers else []

# Load papers into session state
if "papers_list" not in st.session_state:
    st.session_state.papers_list = fetch_papers()

papers = st.session_state.papers_list

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📄 ML Paper RAG")
    
    health = api_get("/health") or {"total_papers": 0, "total_chunks": 0, "model": "Unknown"}
    
    st.metric("Total Papers", health.get("total_papers", 0))
    st.metric("Total Chunks", health.get("total_chunks", 0))
    st.metric("Model", health.get("model", "Unknown"))
    
    st.divider()
    
    st.markdown("### Upload New Paper")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("Ingest Paper"):
            with st.spinner("Processing PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = api_post("/ingest", files=files)
                if res and "chunks_added" in res:
                    st.success(f"Paper ingested! {res['chunks_added']} chunks added.")
                    fetch_papers.clear() # Clear cache
                    st.session_state.papers_list = fetch_papers() # Refresh local state
    
    st.divider()
    st.selectbox("Embedding Model", ["all-mpnet-base-v2"], disabled=True)
    st.caption("Built with FastAPI + ChromaDB + Groq")


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📄 Ask Papers", "⚖️ Compare", "🎯 Interview Prep", "📚 Browse"])

# TAB 1: Ask Papers
with tab1:
    st.header("Ask Questions about ML Papers")
    
    paper_options = {"All Papers": None}
    for p in papers:
        paper_options[p["title"]] = p["id"]
        
    selected_paper_title = st.selectbox("Select Paper", list(paper_options.keys()))
    mode = st.radio("Response Mode", ["Standard", "Beginner-Friendly"], horizontal=True)
    
    question = st.text_area("Ask a question about the paper...")
    
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                payload = {
                    "question": question,
                    "mode": "beginner" if mode == "Beginner-Friendly" else "standard",
                    "paper_id": paper_options[selected_paper_title]
                }
                res = api_post("/query", json=payload)
                if res:
                    st.session_state.last_answer = res.get("answer", "")
                    st.session_state.last_sources = res.get("sources", [])
    
    if "last_answer" in st.session_state:
        with st.container(border=True):
            st.markdown(st.session_state.last_answer)
            
        sources = st.session_state.last_sources
        if sources:
            with st.expander(f"📎 Sources ({len(sources)} chunks)"):
                for s in sources:
                    st.caption(f"{s.get('title', 'Unknown')} — {s.get('section', 'Unknown')}")
                    sim = max(0.0, min(1.0, float(s.get("similarity", 0.0))))
                    st.progress(sim, text=f"Similarity: {sim:.2f}")
                    st.text(s.get("text", "")[:200] + "...")
                    st.divider()

# TAB 2: Compare Papers
with tab2:
    st.header("Compare Papers")
    if len(papers) < 2:
        st.warning("Need at least 2 papers to compare. Please ingest more papers.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            p1_title = st.selectbox("Paper A", [p["title"] for p in papers], key="p1")
        with col2:
            p2_title = st.selectbox("Paper B", [p["title"] for p in papers], key="p2")
            
        aspect = st.text_input("Comparison aspect", placeholder="methodology, results, architecture, training data...")
        
        if st.button("Compare"):
            if p1_title == p2_title:
                st.warning("Please select two different papers.")
            elif aspect:
                p1_id = next(p["id"] for p in papers if p["title"] == p1_title)
                p2_id = next(p["id"] for p in papers if p["title"] == p2_title)
                
                with st.spinner(f"Comparing {p1_title} and {p2_title}..."):
                    payload = {"paper_id_1": p1_id, "paper_id_2": p2_id, "aspect": aspect}
                    res = api_post("/compare", json=payload)
                    if res:
                        st.markdown(res.get("comparison_table", ""))
                        st.info(res.get("summary", ""))

# TAB 3: Interview Prep
with tab3:
    st.header("Generate Interview Questions")
    if not papers:
        st.warning("No papers available.")
    else:
        prep_title = st.selectbox("Select Paper to prepare for", [p["title"] for p in papers])
        if st.button("Generate Questions"):
            p_id = next(p["id"] for p in papers if p["title"] == prep_title)
            with st.spinner("Generating interview questions..."):
                res = api_post("/interview", json={"paper_id": p_id})
                if res:
                    st.session_state.interview_questions = res
                    
    if "interview_questions" in st.session_state:
        iq = st.session_state.interview_questions
        
        conceptual = iq.get("conceptual", [])
        if conceptual:
            with st.expander(f"💡 Conceptual Questions ({len(conceptual)})"):
                for i, q in enumerate(conceptual, 1):
                    st.markdown(f"**Q{i}: {q['question']}**")
                    with st.expander("Show Model Answer", expanded=False):
                        st.markdown(q['model_answer'])
                    st.divider()
                
        implementation = iq.get("implementation", [])
        if implementation:
            with st.expander(f"⚙️ Implementation Questions ({len(implementation)})"):
                for i, q in enumerate(implementation, 1):
                    st.markdown(f"**Q{i}: {q['question']}**")
                    with st.expander("Show Model Answer", expanded=False):
                        st.markdown(q['model_answer'])
                    st.divider()
                
        critical = iq.get("critical", [])
        if critical:
            with st.expander(f"🔍 Critical Thinking Questions ({len(critical)})"):
                for i, q in enumerate(critical, 1):
                    st.markdown(f"**Q{i}: {q['question']}**")
                    with st.expander("Show Model Answer", expanded=False):
                        st.markdown(q['model_answer'])
                    st.divider()

# TAB 4: Browse Papers
with tab4:
    st.header("Browse Indexed Papers")
    
    search_query = st.text_input("Filter papers...", key="paper_filter")
    
    filtered_papers = papers
    if search_query:
        search_lower = search_query.lower()
        filtered_papers = [p for p in papers if search_lower in p["title"].lower() or any(search_lower in a.lower() for a in p.get("authors", []))]
        
    @st.dialog("Paper Summary")
    def show_summary(paper_id):
        with st.spinner("Fetching summary..."):
            res = api_get(f"/paper/{paper_id}/summary")
            if res:
                st.markdown(f"**Summary**\n{res.get('summary', '')}")
                st.markdown("**Key Contributions**")
                for c in res.get("contributions", []):
                    st.markdown(f"- {c}")

    if not filtered_papers:
        st.info("No papers match your filter.")
    else:
        cols = st.columns(3)
        for i, p in enumerate(filtered_papers):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{p['title']}**")
                    authors_str = p.get('authors', ['Unknown'])[0] + " et al." if p.get('authors') else "Unknown authors"
                    st.caption(f"{authors_str} • {p.get('date', '')}")
                    
                    abstract = p.get('abstract', '')
                    if len(abstract) > 200:
                        st.markdown(abstract[:200] + "...")
                    else:
                        st.markdown(abstract)
                        
                    if st.button("View Summary", key=f"summary_{p['id']}"):
                        show_summary(p['id'])
