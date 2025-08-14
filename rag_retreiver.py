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

faiss_path = "faiss_ticket_index"

if os.path.exists(faiss_path):
    vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_texts(texts, embeddings)
    vector_store.save_local(faiss_path)


retriever = vector_store.as_retriever()

qa_chain=RetreivalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 3. Test it
while True:
    user_query=input("Ask your Question (or type exit to quit)")
    if user_query.lower()=="exit":
        break

    result = qa_chain.invoke({"query":user_query})
    print("\nAnswer:", result["result"])
    print("Sources:", result["source_documents"])        