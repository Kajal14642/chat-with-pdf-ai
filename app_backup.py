import streamlit as st
from pypdf import PdfReader
import ollama

st.title("Chat with PDF using AI")
if "messages" not in st.session_state:
    st.session_state.messages = []
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    st.success("PDF uploaded successfully!")

    st.subheader("PDF Content")
    st.write(text[:1000])

    st.write(f"Total characters extracted: {len(text)}")

    question = st.text_input("Ask a question about the PDF")

    if question:

        with st.spinner("Thinking..."):

            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
Here is the PDF content:

{text}

Answer the following question based only on the PDF:

{question}
"""
                    }
                ]
            )

        st.success("Answer")
        answer = response["message"]["content"]

        st.session_state.messages.append(
            {"question": question, "answer": answer}
        )

        st.success("Answer")
        st.write(answer)

        st.subheader("Chat History")

        for chat in reversed(st.session_state.messages):
            st.write("🙋 You:", chat["question"])
            st.write("🤖 AI:", chat["answer"])
            st.write("---")    