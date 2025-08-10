import streamlit as st
import tempfile
import json
import os
from utils import (
    load_reference_corpus,
    build_vector_index,
    retrieve_relevant_passages,
    analyze_documents_with_llm,
    annotate_docx_with_findings,
    DEFAULT_CHECKLIST,
)

st.set_page_config(page_title="ADGM Corporate Agent", layout="wide")
st.title("ADGM-Compliant Corporate Agent — Streamlit Demo")

# --- Auto-build reference index on startup ---
ref_path = "resources"
if "corpus" not in st.session_state:
    if os.path.exists(ref_path):
        corpus = load_reference_corpus(ref_path)
        if corpus:
            index = build_vector_index(corpus)
            st.session_state["corpus"] = corpus
            st.session_state["index"] = index
            st.success(f"Reference index loaded with {len(corpus)} passages.")
        else:
            st.warning("No reference text found in 'resources/'. Using empty index for now.")
            st.session_state["corpus"] = []
            st.session_state["index"] = None
    else:
        st.warning("'resources/' folder not found. Using empty index.")
        st.session_state["corpus"] = []
        st.session_state["index"] = None

# --- Upload files ---
uploaded_files = st.file_uploader("Upload `.docx` files", type=["docx"], accept_multiple_files=True)

if uploaded_files:
    results_summary = []
    output_files = []

    for up in uploaded_files:
        with st.spinner(f"Processing {up.name}..."):
            tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
            tmp.write(up.read())
            tmp.flush()

            if st.session_state["index"]:
                passages = retrieve_relevant_passages(tmp.name, st.session_state["index"], st.session_state["corpus"], top_k=3)
            else:
                passages = []  # No reference data

            analysis = analyze_documents_with_llm(
                tmp.name,
                passages,
                checklist=DEFAULT_CHECKLIST
            )

            reviewed_path = annotate_docx_with_findings(tmp.name, analysis["issues_found"], out_suffix="_reviewed")
            output_files.append({
                "original": up.name,
                "reviewed_path": reviewed_path,
                "analysis": analysis
            })
            results_summary.append({
                "file": up.name,
                "process": analysis.get("process"),
                "documents_uploaded": analysis.get("documents_uploaded"),
                "required_documents": analysis.get("required_documents"),
                "missing_document": analysis.get("missing_document"),
                "issues_count": len(analysis.get("issues_found", [])),
            })

    st.subheader("Summary Table")
    st.table(results_summary)

    for item in output_files:
        st.markdown(f"### {item['original']}")
        st.download_button(
            "Download Reviewed DOCX",
            data=open(item["reviewed_path"], "rb").read(),
            file_name=f"{item['original'][:-5]}_reviewed.docx"
        )
        st.download_button(
            "Download JSON Report",
            data=json.dumps(item["analysis"], indent=2),
            file_name=f"{item['original'][:-5]}_analysis.json"
        )
else:
    st.info("Upload `.docx` files above to start analysis.")
