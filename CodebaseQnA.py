import os
import tempfile
import shutil
import subprocess
from flask import Flask, request
from dotenv import load_dotenv
from getpass import getpass
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)

# Clone the GitHub repo to a temporary directory
repo_url = os.environ.get('REPO_URL')
if not repo_url:
    raise ValueError('REPO_URL environment variable is not set.')
temp_dir = tempfile.mkdtemp()
subprocess.check_call(['git', 'clone', repo_url, temp_dir])

# Load all repository files
root_dir = temp_dir
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.py') and '/.venv/' not in dirpath:
            try: 
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e: 
                pass
print(f'{len(docs)} files loaded.')

# Chunk the files
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(f"{len(texts)} chunks created.")

# Embed chunks and upload them to the DeepLake
embeddings = OpenAIEmbeddings()
db = DeepLake.from_documents(texts, embeddings, dataset_path=f"hub://{os.environ['ACTIVELOOP_TOKEN']}/langchain-code")

# Load the dataset, construct the retriever, and the Conversational Chain
db = DeepLake(dataset_path=f"hub://{os.environ['ACTIVELOOP_TOKEN']}/langchain-code", read_only=True, embedding_function=embeddings)
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 20
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 20

model = ChatOpenAI(model_name='gpt-3.5-turbo') 
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

# At the end of your script, delete the temporary directory
shutil.rmtree(temp_dir)

@app.route('/get_answer', methods=['POST'])
def get_answer_route():
    repo_url = request.form.get('repo_url')
    question = request.form.get('question')
    answer = get_answer(question, repo_url)
    return {'answer': answer}

def get_answer(question, repo_url):
    temp_dir = tempfile.mkdtemp()
    subprocess.check_call(['git', 'clone', repo_url, temp_dir])
    root_dir = temp_dir
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.py') and '/.venv/' not in dirpath:
                try: 
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    docs.extend(loader.load_and_split())
                except Exception as e: 
                    pass
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    db = DeepLake.from_documents(texts, embeddings, dataset_path=f"hub://{os.environ['ACTIVELOOP_TOKEN']}/langchain-code")
    db = DeepLake(dataset_path=f"hub://{os.environ['ACTIVELOOP_TOKEN']}/langchain-code", read_only=True, embedding_function=embeddings)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 20
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 20
    model = ChatOpenAI(model_name='gpt-3.5-turbo') 
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
    chat_history = []
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    shutil.rmtree(temp_dir)
    return result['answer']