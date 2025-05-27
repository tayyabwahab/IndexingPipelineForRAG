from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_data(file = ''):
    
    if file == '':
        print("Provide file name!!!")
        return None
    print('Loading Data...')
    doc_type = file[-3:]

    if doc_type.lower() == 'txt':
        data = TextLoader(file)
    elif doc_type.lower() == 'pdf':
        data = PyPDFLoader(file)
    else:
        print('Wrong input file!!!')
        return None
    
    return data.load()

def split_data(data = ''):
    if data == '':
        print("No data provided!!!")
        return None
    
    print('Splitting Data...')

    split_text = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    data = split_text.split_documents(data)

    return data

def create_store_embeddings(data = ''):
    if data == '':
        print("No data provided for embeddings!!!")
        return None

    print('Creating Embeddings...')
    embedding = GPT4AllEmbeddings() # Replace embedding method here

    vec_db = FAISS.from_documents(data, embedding) 

    print('Embeddings successfully stored into FAISS index.')
    return vec_db

def data_to_vector(file):
    data = load_data(file)
    count = 0
    #print(len(data[0].page_content))
    for d in data:
        count+= len(d.page_content)
    
    if count > 3000:
        data = split_data(data)

    vec_db = create_store_embeddings(data)

    return vec_db


if __name__ == "__main__":
    file = 'file.pdf'           # Replace file name here
    vdb = data_to_vector(file) 
    