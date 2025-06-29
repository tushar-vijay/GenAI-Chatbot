from PyPDF2 import PdfReader  # For reading and extracting text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To break long text into smaller chunks
from langchain.embeddings.openai import OpenAIEmbeddings  # To convert text into numerical vectors using OpenAI
from langchain.vectorstores import FAISS  # To store and search through text vectors efficiently
from langchain.chains.question_answering import load_qa_chain  # To load a question-answering processing chain
from langchain_community.chat_models import ChatOpenAI  # To access OpenAI's GPT model via LangChain

OPENAI_API_KEY = "sk-proj-..."  # OpenAI API key to authenticate GPT model access

# Create the Streamlit UI
st.header("My First Chatbot")  # Title shown at the top of the app

with st.sidebar:
    st.title("Your Documents")  # Sidebar title
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")  # File uploader for PDFs

# When a PDF is uploaded
if file is not None:
    pdf_reader = PdfReader(file)  # Read the uploaded PDF
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()  # Extract and combine text from each PDF page

    # Split the combined text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n",  # Split by newlines
        chunk_size = 1000,  # Each chunk will have max 1000 characters
        chunk_overlap = 150,  # Overlap between chunks to preserve context
        length_function = len  # Use length in characters
    )
    chunks = text_splitter.split_text(text)  # Get the list of text chunks

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Convert chunks into embeddings

    vector_store = FAISS.from_texts(chunks, embeddings)  # Store embeddings in FAISS for fast search

    user_question = st.text_input("Ask a question about the document")  # Text input for user to ask a question

    # When the user asks a question
    if user_question:
        matching_chunks = vector_store.similarity_search(user_question)  # Find chunks similar to the question

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,  # Authenticate with OpenAI
            model_name="gpt-3.5-turbo",  # Use GPT-3.5 Turbo model
            temperature=0,  # Make responses more consistent and less random
            max_tokens=1000  # Limit response length
        )
        
        #chain -> take the ques -> get relevant document -> pass it to LLM -> get answer
        chain = load_qa_chain(llm, chain_type="stuff")  # Load QA chain to process input and produce answers
        response = chain.run(input_documents=matching_chunks, question=user_question)  # Get the final answer
        st.write(response)  # Display the answer in the app
