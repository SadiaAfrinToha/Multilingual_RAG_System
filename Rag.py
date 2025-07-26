# Upload & Install dependencies
from google.colab import files
uploaded = files.upload()

!pip install -q langchain-community cassio datasets langchain openai tiktoken PyPDF2
!pip install -q transformers

# PDF Reading
from PyPDF2 import PdfReader
pdfreader = PdfReader('HSC26-Bangla1st-Paper.pdf')
raw_text = ''
for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# Load HuggingFace Model for LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

# Astra + Cassandra DB Setup
import cassio
ASTRA_DB_APPLICATION_TOKEN = "your_astra_token"
ASTRA_DB_ID = "your_db_id"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Embeddings + Vector Store Init
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Text Split + Store
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n", chunk_size=800, chunk_overlap=200, length_function=len
)
texts = text_splitter.split_text(raw_text)
astra_vector_store.add_texts(texts[:50])
print(f"Inserted {len(texts[:50])} chunks into vector store.")

# Setup DB session
from cassio.config import get_driver_session
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
import uuid
from datetime import datetime

session = get_driver_session()

# Create table if not already present (id UUID, question TEXT, answer TEXT, timestamp TIMESTAMP)
session.execute("""
    CREATE TABLE IF NOT EXISTS rag_bangla_db.qa_logs (
        id UUID PRIMARY KEY,
        question TEXT,
        answer TEXT,
        timestamp TIMESTAMP
    );
""")

# Start interactive QA loop and log to DB
first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if query_text.lower() == "quit":
        break
    if query_text == "":
        continue

    first_question = False

    print(f"\nQUESTION: \"{query_text}\"")
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    print(f"ANSWER: \"{answer}\"\n")

    # ✅ Store Q&A in DB
    session.execute("""
        INSERT INTO rag_bangla_db.qa_logs (id, question, answer, timestamp)
        VALUES (%s, %s, %s, %s)
    """, (uuid.uuid4(), query_text, answer, datetime.utcnow()))

    # Show top 4 relevant chunks
    print("TOP DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print(f"    [{score:.4f}] \"{doc.page_content[:84]} ...\"")
