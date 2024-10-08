### Import Section ###
import os
import time
import chainlit as cl
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import CacheBackedEmbeddings
import uuid 
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
#caching
#from langchain_redis import RedisSemanticCache

from dotenv import load_dotenv
load_dotenv()

#Redis Semantic Cache
REDIS_URL = os.getenv("REDIS_URL")
chat_model = ChatOpenAI(model="gpt-4o-mini")
core_embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

collection_name = f"pdf_to_parse_{uuid.uuid4()}"
client = QdrantClient(":memory:")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Typical QDrant Vector Store Set-up
file_path = os.path.abspath(os.path.join('./', '2307.06435v9.pdf' ))

Loader = PyMuPDFLoader
loader = Loader(file_path)
documents = loader.load()
docs = text_splitter.split_documents(documents)
for i, doc in enumerate(docs):
    doc.metadata["source"] = f"source_{i}"

rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
"""

rag_message_list = [
    {"role" : "system", "content" : rag_system_prompt_template},
]

rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_template),
    ("human", rag_user_prompt_template)
])

store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings, store, namespace=core_embeddings.model
)

# Typical QDrant Vector Store Set-up
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=cached_embedder)
vectorstore.add_documents(docs)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})


### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():
    llm_chain  = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt | chat_model
    )
    llm_chain = chat_model #ChatPromptTemplate.from_template('You are a helpful assistant so answer the question. Question: {question}') | chat_model
    cl.user_session.set('app', llm_chain )
    #cl.user_session.set('vectorstore', vectorstore )


### On Message Section ###

async def main(message: cl.Message):
    _app = cl.user_session.get("app")
    user_id = cl.user_session.get('user_id')

    msg = cl.Message(content="")

    _begin = time.perf_counter()
    response = ""
    async for event in _app.astream({"question": message.content, "context": ""}):
                   response += event.content      
                   await msg.stream_token(event.content)
    _end = time.perf_counter()
    await msg.update()
     
    # Add a button for showing logs
    actions = [
        cl.Action(name="More", value=f"Latency: {str(_end - _begin)} seconds", description="Click to view context")
    ]
    
    await cl.Message(
        content="Click the button to see context",
        actions=actions
    ).send()
    
# Function to show logs or metadata
async def show_more(data):
    await cl.Message(content=data).send()
    
# Event handler for the button action
@cl.action_callback("More")
async def handle_show_more(action):
    log_data = action.value
    await show_more(log_data)
