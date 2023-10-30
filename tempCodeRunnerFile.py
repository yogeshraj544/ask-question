import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

with st.sidebar:
    st.title('llm chat app')
    st.markdown('''
                ''')
    add_vertical_space(5)
    st.write('made with Langchain')


def main():
    st.header("Chat with pdf ")
    

    pdf = st.file_uploader("upload your pdf", type = "pdf")
    #st.write(pdf.name)
    if pdf is not None:

        pdf_reader =PdfReader(pdf)
        
        text= ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
        embeddings = OpenAIEmbeddings(openai_api_key="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")


        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb")as f:
                VectorStore = pickle.load(f)
            #st.write('E mbeddings loaded from the disk')
        
        else:
            embeddings = OpenAIEmbeddings()

            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions :")
        #st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
            #st.write(docs)
       # st.write(chunks)

    #    st.write(text)   
