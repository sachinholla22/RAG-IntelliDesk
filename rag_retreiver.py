from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_mistralai import ChatMistralAI
import psycopg2
import os
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_AI_KEY")

# 2. Initialize LLM
llm = ChatMistralAI(api_key=api_key)

# 3. Connect to Postgres
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAMES"),
    user=os.getenv("DB_USERNAMES"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOSTS"),
    port=os.getenv("DB_PORTS")
)
cur = conn.cursor()

# 4. Fetch tickets
cur.execute("SELECT id, title, description FROM ticket")
rows = cur.fetchall()

# Create documents with metadata
texts = [f"{title} {description}" for (_, title, description) in rows]
metadatas = [{"id": id, "title": title} for (id, title, _) in rows]

# 5. Build embeddings + vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_path = "faiss_ticket_index"

if os.path.exists(faiss_path):
    vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vector_store.save_local(faiss_path)

# 6. Create retriever with optional metadata filters
retriever = vector_store.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Test loop
while True:
    user_query = input("Ask your Question (or type exit to quit): ")
    if user_query.lower() == "exit":
        break

    # Example: add metadata filter (e.g. only search in tickets with title "Bug")
    result = qa_chain.invoke({
        "query": user_query,
        "filters": {"title": "Bug"}   # <-- optional filter
    })

    print("\nAnswer:", result["result"])
    print("Sources:")
    for doc in result["source_documents"]:
        print(f"- ID: {doc.metadata.get('id')}, Title: {doc.metadata.get('title')}")
