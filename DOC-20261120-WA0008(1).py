import streamlit as st

from briefit_engine import Summarizer

st.set_page_config(page_title="BriefIt", page_icon="📝", layout="wide")


@st.cache_resource(show_spinner="Loading the summarisation model (first time takes a while)...")
def get_model():
    return Summarizer()


st.title("BriefIt - Abstractive Text Summariser")
st.caption("Paste a long article or upload a .txt file and get a short summary written by the model.")

model = get_model()
st.sidebar.markdown(f"**Model in use:** `{model.name}`")
length = st.sidebar.radio("Summary length", ["Short", "Medium", "Long"], index=1)
st.sidebar.write("Long articles are split into parts, each part is summarised, and then those summaries are summarised again.")

left, right = st.columns(2)

with left:
    st.subheader("Article")
    uploaded = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
    default_text = uploaded.read().decode("utf-8", errors="ignore") if uploaded else ""
    article = st.text_area("Paste the article here", value=default_text, height=380)
    if article.strip():
        st.caption(f"{len(article.split())} words")
    go = st.button("Generate Summary", type="primary")

with right:
    st.subheader("Summary")
    if go:
        if len(article.split()) < 40:
            st.warning("Please enter at least 40 words, very short text cannot be summarised properly.")
        else:
            with st.spinner("Writing the summary..."):
                result = model.summarize(article, length)
            st.write(result["summary"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Original words", result["input_words"])
            c2.metric("Summary words", result["summary_words"])
            c3.metric("Reduced by", f"{result['compression']:.0%}")
            if result["chunks"] > 1:
                st.caption(f"Article was long, so it was processed in {result['chunks']} parts.")
    else:
        st.info("The summary will appear here.")
