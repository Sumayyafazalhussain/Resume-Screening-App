
import streamlit as st
from src.utils.file_loader import extract_text
from src.predict import predict_resume

st.set_page_config(page_title="Resume Screening App")
st.title("Resume Screening App")

uploaded = st.file_uploader("Upload .pdf/.docx/.txt", type=["pdf","docx","txt"])
text_input = st.text_area("Or paste resume text here", height=200)

if st.button("Analyze"):
    text = ""
    if uploaded is not None:
        text = extract_text(uploaded.name, uploaded.read())
    elif text_input.strip():
        text = text_input.strip()
    else:
        st.warning("Please upload or paste text")
        st.stop()
    try:
        label, score, skills = predict_resume(text)
        st.success("Done")
        st.metric("Predicted Role", label)
        st.metric("Fit Score", f"{score}%")
        st.write("Matched Skills:", skills)
        with st.expander("Preview Text"):
            st.text_area("text", value=text[:5000], height=250)
    except Exception as e:
        st.error(f"Error: {e}")
