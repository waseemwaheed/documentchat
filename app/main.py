import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from html_templates import css, user_template, bot_template
from langchain.schema import HumanMessage

def read_pdf_text(pdf_docs): 
    text = ""
    for doc in pdf_docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput():
    if st.session_state.new_prompt:
        user_question = st.session_state.user_question

        st.session_state.chat_history.append({"source":"Human", 
                                            "text":user_question})
        if st.session_state.conversation:
            answer = st.session_state.conversation({"question": user_question})["answer"]
        else:
            answer = "No files are loaded, please load (a) files and click process to give me some context 😊"

        st.session_state.chat_history.append({"source":"AI", 
                                            "text":answer})
        
        # reset the new prompt flag to avoid adding the same question/answer to the chat history
        # this is done to tackle a streamlit quirk 
        st.session_state.new_prompt = False
            
    # display messages in reverse chronological order
    for message in st.session_state.chat_history[::-1]:
            if message["source"] == "Human":
                st.write(user_template.format(MSG=message["text"]), unsafe_allow_html=True)
            else:
                st.write(bot_template.format(MSG=message["text"]), unsafe_allow_html=True)
    
def user_question_submit():
    st.session_state.user_question = st.session_state.prompt
    st.session_state.new_prompt = True
    st.session_state.prompt = ""

def main():

    # load secrets
    load_dotenv()

    # page config
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    # add custom bot CSS
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "new_prompt" not in st.session_state:
        st.session_state.new_prompt = False

    # header
    st.header("chat with multiple PDFs :books:")

    # textbox
    st.text_input("Ask a question:", key="prompt", on_change=user_question_submit)
    
    # handle user prompts
    if "user_question" in st.session_state:
        handle_userinput()

    # sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("upload your PDFs here and click `Process`", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):

                # read file(s)
                content = read_pdf_text(pdf_docs)
                st.write("Text extracted")
                
                # chunk text
                chunks = get_text_chunks(content)
                st.write("Text chuncked")

                # create a vector store
                vectorstore = get_vectorstore(chunks)
                st.write("Text vectorized")

                # instantiate a conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.write("Conversation chain initialised")
        

if __name__ == "__main__":
    main()