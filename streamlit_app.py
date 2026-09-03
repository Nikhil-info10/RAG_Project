import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / "Code" / ".env")

try:
    import retrieval_and_rag as rag
except Exception as exc:
    st.error(f"Could not load the retrieval system: {exc}")
    st.stop()


st.set_page_config(
    page_title="IIT PATNA:InnoCorp Knowledge Assistant",
    page_icon="K",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading search index...")
def load_search_resources():
    embeddings = rag.HuggingFaceEmbeddings(model_name=rag.HF_MODEL_NAME)
    database = rag.Chroma(persist_directory=rag.CHROMA_DIR, embedding_function=embeddings)
    stored_docs = database.get(include=["metadatas", "documents"])
    documents = stored_docs.get("documents", [])
    metadatas = stored_docs.get("metadatas", [])
    bm25_documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(documents, metadatas)
    ]
    if not bm25_documents:
        raise RuntimeError("The Chroma database is empty. Run ingest.py first.")
    bm25 = rag.BM25Retriever.from_documents(bm25_documents)
    records = rag.load_employee_records()
    names = [str(record["Full Name"]) for record in records]
    return database, bm25, records, names


def source_documents(docs: List[Any]) -> List[str]:
    return rag.source_names(docs)


def answer_question(
    question: str,
    database: Any,
    bm25: Any,
    records: List[Dict[str, Any]],
    names: List[str],
    history: List[Dict[str, Any]],
):
    search_query = rag.build_search_query(question, history)
    vector_results = database.similarity_search(search_query, k=10)
    keyword_results = bm25.invoke(search_query)
    reranked = rag.rerank_results(search_query, vector_results + keyword_results)

    current_query = question.lower()
    is_policy_question = any(term in current_query for term in rag.POLICY_TERMS)
    has_unrelated_intent = any(term in current_query for term in rag.NON_EMPLOYEE_TERMS)
    is_employee_follow_up = not has_unrelated_intent and (
        rag.is_employee_question(question) or any(
            rag.is_employee_question(turn.get("content", ""))
            for turn in history
            if turn.get("role") == "user"
        )
    )
    employee_answer = None
    if not is_policy_question:
        employee_answer = rag.answer_from_employee_data(
            search_query,
            reranked,
            names,
            records,
            current_question=question,
        )
    if employee_answer:
        return employee_answer, ["InnoCorp_Solutions_Employee_Details.xlsx"]

    context_docs = reranked
    if is_policy_question or not is_employee_follow_up:
        context_docs = [
            doc
            for doc in reranked
            if "Employee_Details.xlsx" not in str(rag.document_metadata(doc).get("source", ""))
        ]
    if not context_docs:
        if is_employee_follow_up:
            return rag.NOT_FOUND_MESSAGE, []
        answer = rag.answer_from_external_search(question)
        if answer.startswith(("External search failed", "No LLM provider", "[LLM call failed:")) or answer == "No web results found.":
            raise RuntimeError(answer)
        return answer, ["Web search"]

    context = rag.assemble_context(context_docs, rag.MAX_CONTEXT_CHARS)
    answer = rag.call_llm(rag.build_prompt(context, search_query, history))
    if rag.needs_external_search(answer):
        if is_employee_follow_up:
            return rag.NOT_FOUND_MESSAGE, source_documents(context_docs)
        answer = rag.answer_from_external_search(question)
        if answer.startswith(("External search failed", "No LLM provider", "[LLM call failed:")) or answer == "No web results found.":
            raise RuntimeError(answer)
        return answer, ["Web search"]
    return rag.add_sources(answer, context_docs), source_documents(context_docs)


st.title("IIT PATNA:InnoCorp Knowledge Assistant")
st.caption("Search company policies, employee records, and internal documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Conversation")
    st.write(f"{len(st.session_state.messages) // 2} question(s) asked")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Answers are grounded in retrieved workspace documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

question = st.chat_input("Ask a question about the company knowledge base...")
if question:
    history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            database, bm25, records, names = load_search_resources()
            with st.spinner("Searching internal documents..."):
                answer, sources = answer_question(
                    question,
                    database,
                    bm25,
                    records,
                    names,
                    history,
                )
            st.markdown(answer)
            if sources:
                with st.expander("Sources", expanded=True):
                    for source in sources:
                        st.markdown(f"- `{source}`")
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        except Exception as exc:
            message = f"I couldn't answer that question: {exc}"
            st.error(message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": message,
                "sources": [],
            })
