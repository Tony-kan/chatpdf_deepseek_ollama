import streamlit as st
import main as main
# from main import pdf_directory,create_vector_store


st.title("Chat with PDF with Deepseek")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf",accept_multiple_files=False)

if uploaded_file:
    main.upload_pdf(uploaded_file)
    db = main.create_vector_store(main.pdf_directory + uploaded_file.name)
    question = st.chat_input("Ask a question")
    if question:
        st.chat_message("user").write(question)
        related_documents = main.retrieve_docs(db,question)
        answer = main.question_pdf(question,related_documents)
        st.chat_message("assistant").write(answer)
        # st.text(main.retrieve_docs(db=db,query="passing score"))