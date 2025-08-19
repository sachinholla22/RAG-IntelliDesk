# file: rag_ticket_ingest_and_query.py

import os
import psycopg2
from dotenv import load_dotenv
from textwrap import shorten

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app=FastApi()
# --- DB connection ---
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAMES"),
    user=os.getenv("DB_USERNAMES"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOSTS"),
    port=os.getenv("DB_PORTS"),
)
conn.autocommit = True

# --- Models ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatMistralAI(api_key=os.getenv("MISTRAL_AI_KEY"))

# ==============
# 1) FETCH DATA
# ==============
def fetch_ticket_docs_for_org(org_id: int, limit_comments=3):
    """
    Returns a list of (text, metadata) tuples for the given org_id.
    Each item is one Ticket with condensed recent comments.
    """
    with conn.cursor() as cur:
        # Tickets + client/assignee names
        cur.execute("""
            SELECT
                t.id AS ticket_id,
                t.title,
                COALESCE(t.description, '') AS description,
                t.status,
                t.priority,
                t.created_at,
                t.due_date,
                c.name AS client_name,
                a.name AS assigned_to_name
                
            FROM ticket t
            LEFT JOIN users c ON c.id = t.client_id
            LEFT JOIN users a ON a.id = t.assigned_to_id
           
            WHERE t.org_id = %s
            ORDER BY t.id DESC
        """, (org_id,))
        tickets = cur.fetchall()

        # Build a map: ticket_id -> recent comments (latest first)
        cur.execute("""
            SELECT cm.ticket_id, u.name AS commenter, cm.comment, cm.last_updated
            FROM comments cm
            LEFT JOIN users u ON u.id = cm.user_id
            WHERE cm.ticket_id IN (
                SELECT id FROM ticket WHERE org_id = %s
            )
            ORDER BY cm.last_updated DESC
        """, (org_id,))
        comments_rows = cur.fetchall()

    comments_by_ticket = {}
    for ticket_id, commenter, comment, last_updated in comments_rows:
        comments_by_ticket.setdefault(ticket_id, [])
        comments_by_ticket[ticket_id].append(f"{commenter}: {comment}")

    docs = []
    for (ticket_id, title, description, status, priority, created_at, due_date,
         client_name, assigned_to_name) in tickets:

        recent_comments = comments_by_ticket.get(ticket_id, [])[:limit_comments]
        comments_text = " | ".join(recent_comments) if recent_comments else "No recent comments."

        # Keep text concise but informative (helps small-context models)
        description_snip = shorten(description, width=400, placeholder="...")

        text = (
            f"Ticket #{ticket_id} ({status}, {priority}) — {title}.\n"
            f"Description: {description_snip}\n"
            f"Client: {client_name or 'N/A'} | AssignedTo: {assigned_to_name or 'Unassigned'}\n"
            f"Recent comments: {comments_text}\n"
            f"CreatedAt: {created_at} | DueDate: {due_date or 'N/A'}\n"
            f"Organization ID: {org_id}"
        )

        metadata = {
            "ticket_id": ticket_id,
            "org_id": org_id,
            "status": status,
            "priority": str(priority),
            "assigned_to": assigned_to_name or "",
            "client": client_name or "",
            "title": title,
        }

        docs.append((text, metadata))

    return docs

# =========================
# 2) BUILD / LOAD INDEXES
# =========================
def upsert_faiss_for_org(org_id: int):
    """
    Create or update FAISS index for a given org_id from DB.
    """
    index_dir = f"indexes/org_{org_id}"

    # Fetch docs
    docs = fetch_ticket_docs_for_org(org_id)
    texts = [t for (t, m) in docs]
    metadatas = [m for (t, m) in docs]

    if os.path.exists(index_dir):
        # Load and add new docs (simplest approach: rebuild; for prod, track upserts)
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        vs.add_texts(texts=texts, metadatas=metadatas)
        vs.save_local(index_dir)
    else:
        vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        vs.save_local(index_dir)

    return index_dir

# ======================================
# 3) QUERY (PER ORG) WITH CUSTOM PROMPT
# ======================================
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an assistant for a ticketing system. Answer ONLY using the context.\n"
        "If the context is not relevant, say you don't have info.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer clearly and concisely."
    ),
)

def ask_for_org(org_id: int, question: str, role: str, user_id: int):
    index_dir = f"indexes/org_{org_id}"
    if not os.path.exists(index_dir):
        raise RuntimeError(f"No index for org_id={org_id}. Run upsert_faiss_for_org first.")

    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    def metadata_filter(metadata):
        if metadata.get("visibility")=="all":
            return True
        if metadata.get("visibility")==role:
            return True
        if metadata.get("created_by")==user_id:
            return True
        return False    
            
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4},
        search_type="similarity",
        filter=metadata_filter
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    result = qa.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [d.page_content[:300] for d in result.get("source_documents", [])]
    }

class RAGRequest(BaseModel):
    orgId: int
    userId: int
    role: str
    question: str


@app.post("/ask")
def ask(request:RAGRequest):
    result = ask_for_org(request.orgId,request.question)
    return {"answer": result["answer"], "sources":result["sources"]}