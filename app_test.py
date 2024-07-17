import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

def extract_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(pdf_text)
    return chunks


def create_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(
        texts=text_chunks, embedding=OpenAIEmbeddings()
    )

    return vectorstore


def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-0125"
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever(),
    )
    return conversation_chain


def handle_user_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user', avatar_style="initials", seed="Kavita")
        else:
            message(msg.content, is_user=False, key=str(i), avatar_style="initials", seed="AI",)



def main():
    load_dotenv()
    st.set_page_config(page_title='ChatBot', page_icon=':books:')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Ask questions from Knowledge Base :books:')

    user_question = st.text_input('Ask a question about your documents:')
    if user_question:
        handle_user_question(user_question)


    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader('Upload a PDF and click process', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):
            
                # get pdf text
                pdf_text = extract_text_from_pdf(pdf_docs) 

                # get the text chunks
                text_chunks = get_text_chunks(pdf_text)
                # st.write(text_chunks)

                # create vector store
                vectorstore = create_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = create_conversation_chain(vectorstore)


            st.success('Processed successfully')




if __name__ == '__main__':
    main()