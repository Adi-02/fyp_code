from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import hashlib
import os
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key")
GROQ_MODEL_NAME = "llama3-8b-8192"

@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class VectorStoreManager:
    def __init__(self):
        self.ticker_news_vector_stores = {}
        self.twitter_vector_stores = {}
        self.general_news_vector_store = None
        self.embedding_model = get_embedding_model()

    def get_or_create_vector_store(self, ticker, source):
        if source == "general_news":
            if self.general_news_vector_store is None:
                self.general_news_vector_store = Chroma(
                    collection_name="general_news",
                    embedding_function=self.embedding_model,
                    persist_directory="./chroma_db/general_news"
                )
            return self.general_news_vector_store

        elif source == "news":
            clean_ticker = self._clean_ticker(ticker)
            if clean_ticker not in self.ticker_news_vector_stores:
                vector_store = Chroma(
                    collection_name=f"{clean_ticker}_news",
                    embedding_function=self.embedding_model,
                    persist_directory=f"./chroma_db/{clean_ticker}_news"
                )
                self.ticker_news_vector_stores[clean_ticker] = vector_store
            return self.ticker_news_vector_stores[clean_ticker]

        elif source == "twitter":
            clean_ticker = self._clean_ticker(ticker)
            if clean_ticker not in self.twitter_vector_stores:
                vector_store = Chroma(
                    collection_name=f"{clean_ticker}_twitter",
                    embedding_function=self.embedding_model,
                    persist_directory=f"./chroma_db/{clean_ticker}_twitter"
                )
                self.twitter_vector_stores[clean_ticker] = vector_store
            return self.twitter_vector_stores[clean_ticker]

        else:
            raise ValueError("Invalid data_type. Use 'news', 'general_news', or 'twitter'.")

    def _clean_ticker(self, ticker):
        if ticker is None:
            return "general"
        clean = ticker.replace('.', '_').replace('-', '_')
        return clean if len(clean) >= 3 else clean + "_stock"


def generate_id(text, date):
    return hashlib.md5(f"{text}_{date}".encode("utf-8")).hexdigest()

def store_text(manager, ticker, texts, source, date):
    vector_store = manager.get_or_create_vector_store(ticker, source)

    new_texts = []
    new_metadatas = []
    new_ids = []

    for text in texts:
        text_id = generate_id(text, date) 
        new_texts.append(text)
        new_metadatas.append({"source": source, "date": date})
        new_ids.append(text_id)

    if not new_texts:
        print(f"No new texts to add for {ticker} in {source} DB")
        return

    try:
        vector_store.add_texts(texts=new_texts, metadatas=new_metadatas, ids=new_ids)
        print(f"Stored {len(new_texts)} new texts for {ticker if ticker else 'general news'} in {source} DB")

    except Chroma.errors.DuplicateIDError as e:
        print(f"DuplicateIDError detected: {e}")
        import re
        duplicate_ids = re.findall(r"[a-f0-9]{32}", str(e))
        if duplicate_ids:
            print(f"Removing duplicate IDs: {duplicate_ids}")
            try:
                vector_store.delete(ids=duplicate_ids)
                print("Retrying insertion after deletion...")
                vector_store.add_texts(texts=new_texts, metadatas=new_metadatas, ids=new_ids)
                print(f"Stored texts after resolving duplicates for {ticker} in {source} DB")
            except Exception as inner_e:
                print(f"Failed again even after deletion: {inner_e}")
        else:
            print("Duplicate ID not identifiable from error message.")

def generate_sentiment_score(news_texts, general_news_texts, tweet_texts):
    combined_context = ""

    if news_texts:
        combined_context += "\n".join([f"[Stock News {i+1}]: {t}" for i, t in enumerate(news_texts)]) + "\n"
    if general_news_texts:
        combined_context += "\n".join([f"[General News {i+1}]: {t}" for i, t in enumerate(general_news_texts)]) + "\n"
    if tweet_texts:
        combined_context += "\n".join([f"[Tweet {i+1}]: {t}" for i, t in enumerate(tweet_texts)]) + "\n"

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=0.5
    )

    prompt = PromptTemplate(
        template="""
        You are a financial sentiment analyst.
        Given the following recent news, market insights, and tweets:

        {context}

        Output a single sentiment score between -1 (very negative) and 1 (very positive) reflecting the market mood.

        Only output the number:
        """,
        input_variables=["context"]
    )

    chain = prompt | llm

    while True:
        output = chain.invoke({"context": combined_context}).content.strip()
        print(f"LLM Output: {output}")
        try:
            return float(output)
        except ValueError:
            print("Invalid output. Retrying...")

def run_rag_pipeline_for_ticker_to_generate_score(manager, ticker, date, new_info):
    if new_info["news"] is None:
        return None

    store_text(manager, ticker, new_info["news"], "news", date)
    if new_info["general_news"] is not None:
        store_text(manager, None, new_info["general_news"], "general_news", date)
    if new_info["twitter"] is not None:
        store_text(manager, ticker, new_info["twitter"], "twitter", date)

    news_docs = manager.get_or_create_vector_store(ticker, "news").get(where={"date": date})
    news_texts = news_docs.get("documents", []) if news_docs else []

    if new_info["general_news"] is not None:
        general_docs = manager.get_or_create_vector_store(None, "general_news").get(where={"date": date})
        general_news_texts = general_docs.get("documents", []) if general_docs else []
    else:
        general_news_texts = None

    if new_info["twitter"] is not None:
        twitter_docs = manager.get_or_create_vector_store(ticker, "twitter").get(where={"date": date})
        twitter_texts = twitter_docs.get("documents", []) if twitter_docs else []
    else:
        twitter_texts = None

    return generate_sentiment_score(news_texts, general_news_texts, twitter_texts)

def rag_conversation_query(manager, ticker, user_query, top_k=5, include=False):
    news_store = manager.get_or_create_vector_store(ticker, "news")
    news_docs = news_store.similarity_search(user_query, k=top_k)

    twitter_docs = []
    general_docs = []

    if include:
        twitter_store = manager.get_or_create_vector_store(ticker, "twitter")
        general_store = manager.get_or_create_vector_store(None, "general_news")
        twitter_docs = twitter_store.similarity_search(user_query, k=top_k)
        general_docs = general_store.similarity_search(user_query, k=top_k)

    context = ""
    if news_docs:
        context += "\n".join([f"[Stock News]: {doc.page_content}" for doc in news_docs]) + "\n"
    if twitter_docs:
        context += "\n".join([f"[Tweet]: {doc.page_content}" for doc in twitter_docs]) + "\n"
    if general_docs:
        context += "\n".join([f"[General News]: {doc.page_content}" for doc in general_docs]) + "\n"

    if not context.strip():
        return "Sorry, I couldn't find anything relevant to your question."

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=0.7,
    )

    prompt = PromptTemplate(
        template="""
        You are a financial assistant helping an investor understand market sentiment and trends.
        Based on the following context from financial news and social media:

        {context}

        Answer the user's question: "{question}"

        Provide a detailed yet concise response.
        """,
        input_variables=["context", "question"]
    )

    chain = prompt | llm
    return chain.invoke({"context": context, "question": user_query}).content.strip()
