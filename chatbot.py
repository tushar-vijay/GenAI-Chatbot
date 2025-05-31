import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "OPENAI_API_KEY"

#Upload PDF files
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)
        
    #Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n",
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    #Generating Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #Creating a Vector Store
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user question
    user_question = st.text_input("Ask a question about the document")

    #do similarity search
    if user_question:
        matching_chunks = vector_store.similarity_search(user_question)
        #st.write(matching_chunks)

        #define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0, #degree of randomness in the response
            max_tokens=1000  #maximum number of tokens in the response
            )

        #output results
        #chain -> take the ques -> get relevant document -> pass it to LLM -> get answer
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=matching_chunks, question=user_question)
        st.write(response)
