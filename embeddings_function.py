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
# import configuration as cn
import warnings
warnings.filterwarnings('ignore')


def pdf_embed():
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

    def load_docs(directory):
        loader = AzureBlobStorageContainerLoader(directory)
        documents = loader.load()
        return documents

    loader =  AzureBlobStorageContainerLoader(conn_str="DefaultEndpointsProtocol=https;AccountName=newstorage103;AccountKey=ge7eafGR+8JUAFRVGe2NGZMzniLkWfHsfNjI/rlohgs+zcArx3+T+gUXgc6C3qFBdQkmufvCQtEY+AStj3XRtA==;EndpointSuffix=core.windows.net",
        container="data"
    )

    # directory = r"C:\Users\GondesisivaramSantos\Downloads\speech"
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    len(encoding.encode(documents[0].page_content))
    

    return len(docs)




