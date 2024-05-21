import streamlit as st
import fitz  # PyMuPDF
import docx

st.set_page_config(page_title="Document Reader", page_icon="📄")
st.title("📄 Document Reader")

uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf", "docx", "txt"])

def read_pdf(file):
    document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text()
    return text

def read_docx(file):
    document = docx.Document(file)
    text = "\n".join([para.text for para in document.paragraphs])
    return text

def read_txt(file):
    return file.read().decode("utf-8")

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"**Filename:** {uploaded_file.name}")
        
        if uploaded_file.type == "application/pdf":
            text = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = read_docx(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = read_txt(uploaded_file)
        else:
            text = "Unsupported file type."
        
        st.text_area("File Content", text, height=300)

else:
    st.info("Please upload documents to proceed.")
