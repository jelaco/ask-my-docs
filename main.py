import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📄",
    layout="centered"
)

# ==========================================
# STYLING
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    h1 { color: #4F7CFF; }
    .stTextInput input {
        background-color: #161B22;
        color: #E6EAF2;
        border: 1px solid #30363D;
    }
    .stButton button {
        background-color: #238636;
        color: white;
        border: none;
        width: 100%;
    }
    .answer-box {
        background-color: #161B22;
        border: 1px solid #4F7CFF;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        color: #E6EAF2;
    }
    .info-box {
        background-color: #161B22;
        border: 1px solid #3FE0D0;
        border-radius: 8px;
        padding: 0.75rem;
        color: #3FE0D0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# API KEY CHECK
# ==========================================
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not google_api_key:
    st.error("⚠️ GOOGLE_API_KEY missing in .env file!")
    st.stop()

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY missing in .env file!")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["GROQ_API_KEY"] = groq_api_key

# ==========================================
# IMPORTS
# ==========================================
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# HEADER
# ==========================================
st.title("📄 Ask My Docs")
st.caption("Upload a document and ask questions about it.")
st.markdown('<div class="info-box">🌍 Supports all languages – upload documents and ask questions in any language!</div>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# SESSION STATE
# ==========================================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "answer" not in st.session_state:
    st.session_state.answer = None

# ==========================================
# FILE UPLOAD
# ==========================================
st.subheader("1️⃣ Upload Document")

uploaded_file = st.file_uploader(
    "Choose a file (TXT or PDF)",
    type=["txt", "pdf"],
    help="Supported formats: .txt, .pdf"
)

if uploaded_file is not None and not st.session_state.doc_loaded:
    with st.spinner("📚 Processing document..."):
        try:
            # Temp file erstellen
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Loader basierend auf Dateityp
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")
            
            documents = loader.load()
            
            # Text splitten
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
            
            # Embeddings erstellen (Google)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.doc_loaded = True
            
            # Temp file löschen
            os.unlink(tmp_path)
            
            st.success(f"✅ Document loaded! ({len(chunks)} sections created)")
            
        except Exception as e:
            st.error(f"❌ Error loading document: {e}")

# Reset Button
if st.session_state.doc_loaded:
    if st.button("🔄 Load new document"):
        st.session_state.vector_store = None
        st.session_state.doc_loaded = False
        st.session_state.answer = None
        st.rerun()

st.markdown("---")

# ==========================================
# QUESTION INPUT
# ==========================================
st.subheader("2️⃣ Ask a Question")

question = st.text_input(
    "Your question:",
    placeholder="e.g. Who is the department head?",
    disabled=not st.session_state.doc_loaded
)

submit = st.button("🔍 Ask", disabled=not st.session_state.doc_loaded or not question)

# ==========================================
# ANSWER GENERATION
# ==========================================
if submit and question and st.session_state.vector_store:
    with st.spinner("🤔 Searching for answer..."):
        try:
            # Retriever erstellen
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            
            # LLM (Groq - schnell & kostenlos)
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            
            # Strenger Prompt gegen Halluzination
            template = """
STRICT RULES:
1. Answer ONLY using information from the CONTEXT below
2. Do NOT invent or add any information
3. If the answer is not in the context, say ONLY: "This information is not contained in the document."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
            prompt = ChatPromptTemplate.from_template(template)
            
            # Hilfsfunktion: Dokumente zu Text
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # RAG Chain
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Antwort generieren
            st.session_state.answer = rag_chain.invoke(question)
            
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Antwort anzeigen
if st.session_state.answer:
    st.markdown("---")
    st.subheader("💡 Answer")
    st.markdown(f"""
    <div class="answer-box">
        {st.session_state.answer}
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Built with 🧠 Google Embeddings + Groq Llama 3.3 + Streamlit")
