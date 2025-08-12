from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
import psycopg2
import os
from dotenv import load_dotenv


load_dotenv()
api_key=os.getenv("MISTRAL_AI_KEY")
llm=ChatMistralAI(api_key=api_key)
#connect to postgres
conn=psycopg2.connect(
    dbname=os.getenv("DB_NAMES"),
    user=os.getenv("DB_USERNAMES"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOSTS"),
    port=os.getenv("DB_PORTS")
)

cur=conn.cursor()

cur.execute("SELECT description,title FROM ticket")
rows=cur.fetchall()
texts = [f"{title} {description}" for description, title in rows]

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store=FAISS.from_texts(texts,embeddings)

vector_store.save_local("faiss_ticket_index")


retriever = vector_store.as_retriever()

qa_chain=RetreivalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 3. Test it
query = "Show me tickets related to database errors"
result = qa_chain.invoke({"query": query})

print("Answer:", result["result"])
print("Sources:", result["source_documents"])