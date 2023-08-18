import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = "YOUR API KEY HERE"


def main():
    st.set_page_config(page_title='Ask PDF ')
    st.header("Please Upload Your PDF 🧐")

    # Upload PDF
    pdf=st.file_uploader(label="Please Upload Your File Here" ,type="pdf")
    
    # from PyPDF2 import PdfReader

    # reader = PdfReader("example.pdf")
    # number_of_pages = len(reader.pages)
    # page = reader.pages[0]
    # text = page.extract_text()

    # As we have to extract the text as a whole,we are embedding It Into Text String
   

    
   
    if pdf:
         reader = PdfReader(pdf)
         text =""
         for page in reader.pages:
             text =text + page.extract_text()

        #  st.write(text)

        # We can not feed whole Data ofthe PDF and Expect togive answerts by the Lnaguage Model. 
        # So, It Is necessary to divide whole Data Into Chunks and let Language Model decide where our relevent information resides 

        # Split the whole Text Into differennt smaller chunks
         splitter  = CharacterTextSplitter(
            separator='\n',
            chunk_size = 1000,
            chunk_overlap  = 300,
            length_function = len,
            # is_separator_regex = False,

    )

         chunks = splitter.split_text(text)

        #  st.write(chunks)


        # Embeddings

         embeddings_model = OpenAIEmbeddings()
         Knowledge_Base = FAISS.from_texts(chunks, embeddings_model)

        # Ask User Input
         user_Prompt = st.text_input("Ask A Question About Your PDF:")

         # Similarity Search

         if user_Prompt:
             docs =  Knowledge_Base.similarity_search(user_Prompt)

            #  st.write(docs)

             chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
             response = chain.run(input_documents=docs, question=user_Prompt)

             st.write(response)





if __name__=='__main__':
    main()