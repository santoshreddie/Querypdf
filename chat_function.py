import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import AzureSearch
import tiktoken
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import AzureBlobStorageContainerLoader
# from langchain.vectorstores import OpenAIEmbeddings
# from langchain.vectorstores import AzureOpenAI
import json
from langchain.chains import RetrievalQA

def chat_with_pdf(user_input):

    creds = json.load(open(r'configuration.json'))

    openai.api_type = creds["OPENAI_API_TYPE"]
    openai.api_base = creds["OPENAI_API_BASE"]
    openai.api_key = creds["OPENAI_API_KEY"]
    openai.api_version = creds["OPENAI_API_VERSION"]

    llm = AzureOpenAI(
        openai_api_key=creds["OPENAI_API_KEY"],
        openai_api_version=creds["OPENAI_API_VERSION"],
        temperature=0,
        deployment_name="gpt-35-turbo"
    )

    embeddings = OpenAIEmbeddings(
        openai_api_key=creds["OPENAI_API_KEY"],
        openai_api_version=creds["OPENAI_API_VERSION"],
        deployment_id="text-embedding-ada-002",
        chunk_size=1
    )
    

    acs = AzureSearch(
        azure_search_endpoint=creds["AZURE_COGNITIVE_SEARCH_ADDRESS"],
        azure_search_key=creds["AZURE_COGNITIVE_SEARCH_API_KEY"],
        index_name=creds["AZURE_COGNITIVE_SEARCH_INDEX_NAME"],
        embedding_function=embeddings.embed_query
    )

    retriever = acs.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    result = qa.run(user_input)
    return result


