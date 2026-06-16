import streamlit as st
from pypdf import PdfReader
from groq import Groq
import os
client_groq = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------
# PAGE TITLE
# ---------------------------

st.title("Chat with PDF using AI")

# ---------------------------
# SESSION STATE
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# FILE UPLOAD
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

# ---------------------------
# PROCESS PDF
# ---------------------------

if uploaded_file:

    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # ---------------------------
    # CHUNKING
    # ---------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    st.write(f"Number of chunks created: {len(chunks)}")

    # ---------------------------
    # EMBEDDINGS
    # ---------------------------

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(chunks)

    st.write(f"Embeddings created: {len(embeddings)}")

    # ---------------------------
    # CHROMADB
    # ---------------------------

    client = chromadb.Client()

    try:
        client.delete_collection("pdf_collection")
    except:
        pass

    collection = client.create_collection(
        name="pdf_collection"
    )

    for i, chunk in enumerate(chunks):

        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            ids=[str(i)]
        )

    st.success("PDF uploaded successfully!")

    st.write(
        f"Stored {len(chunks)} chunks in ChromaDB"
    )

    # ---------------------------
    # PDF PREVIEW
    # ---------------------------

    st.subheader("PDF Content")

    st.write(text[:1000])

    st.write(
        f"Total characters extracted: {len(text)}"
    )

    # ---------------------------
    # QUESTION INPUT
    # ---------------------------

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        question_embedding = model.encode(
            question
        ).tolist()

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=5
        )

        st.write(
            f"Retrieved {len(results['documents'][0])} relevant chunks"
        )

        context = "\n".join(
            results["documents"][0]
        )

        prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{question}
"""

        # ---------------------------
        # OLLAMA
        # ---------------------------

        with st.spinner("Thinking..."):

            response = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        answer = response.choices[0].message.content    


        # ---------------------------
        # CHAT HISTORY
        # ---------------------------

        st.session_state.messages.append(
            {
                "question": question,
                "answer": answer
            }
        )

        # ---------------------------
        # ANSWER
        # ---------------------------

        st.subheader("Answer")

        st.write(answer)

        # ---------------------------
        # RETRIEVED CHUNKS
        # ---------------------------

        st.subheader("Retrieved Context")

        for i, doc in enumerate(
            results["documents"][0]
        ):

            st.write(f"Chunk {i+1}")

            st.write(doc)

            st.write("---")

        # ---------------------------
        # CHAT HISTORY
        # ---------------------------

        st.subheader("Chat History")

        for chat in reversed(
            st.session_state.messages
        ):

            st.write(
                f"🙋 You: {chat['question']}"
            )

            st.write(
                f"🤖 AI: {chat['answer']}"
            )

            st.write("---")