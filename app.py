import streamlit as st
import requests
import pandas as pd
import time
import re
import warnings
warnings.filterwarnings("ignore")

from io import StringIO
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
import yfinance as yf

# ========== 종목 검색 ==========
def is_korean(text):
    return any("\uAC00" <= c <= "\uD7A3" for c in text)

def get_stock_code(stock_name):
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.encoding = "euc-kr"
    df = pd.read_html(StringIO(response.text))[0]
    df["종목코드"] = df["종목코드"].astype(str).str.strip()
    result = df[df["회사명"] == stock_name]["종목코드"]
    return result.values[0] if not result.empty else None

def translate_to_english(korean_name):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": "ko", "tl": "en", "dt": "t", "q": korean_name}
    try:
        response = requests.get(url, params=params)
        return response.json()[0][0][0]
    except:
        return None

def get_yahoo_ticker(query):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "lang": "en-US", "region": "US"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, params=params, headers=headers)
        quotes = response.json().get("quotes", [])
        return quotes[0]["symbol"] if quotes else None
    except:
        return None

def search_stock(stock_name):
    if stock_name.isdigit() and len(stock_name) == 6:
        return {"type": "KR", "code": stock_name, "name": stock_name}
    
    if is_korean(stock_name):
        code = get_stock_code(stock_name)
        if code:
            return {"type": "KR", "code": code, "name": stock_name}
        english_name = translate_to_english(stock_name)
        if english_name:
            ticker = get_yahoo_ticker(english_name)
            if ticker:
                return {"type": "US", "code": ticker, "name": stock_name}
        return None
    else:
        ticker = get_yahoo_ticker(stock_name)
        if ticker:
            return {"type": "US", "code": ticker, "name": stock_name}
        return None

# ========== 뉴스 크롤링 ==========
def get_article_body(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.naver.com"
    }
    try:
        response = requests.get(url, headers=headers)
        response.encoding = "euc-kr"
        if "top.location.href=" in response.text:
            match = re.search(r"top\.location\.href='(.+?)'", response.text)
            if match:
                real_url = match.group(1)
                response = requests.get(real_url, headers=headers)
                response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        for selector in ["div#dic_area", "div.newsct_article", "div#news_read", "div#contents"]:
            body = soup.select_one(selector)
            if body and body.text.strip():
                return body.text.strip()[:500]
        return ""
    except:
        return ""

def get_naver_finance_news(stock_info, num_articles=20):
    code = stock_info["code"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.naver.com"
    }
    results = []
    existing_links = set()
    page = 1
    while len(results) < num_articles:
        url = f"https://finance.naver.com/item/news_news.naver?code={code}&page={page}"
        response = requests.get(url, headers=headers)
        response.encoding = "euc-kr"
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("td.title a")
        if not articles:
            break
        for article in articles:
            if len(results) >= num_articles:
                break
            title = article.text.strip()
            link = "https://finance.naver.com" + article["href"]
            if link in existing_links:
                continue
            body = get_article_body(link)
            results.append({"title": title, "link": link, "body": body})
            existing_links.add(link)
            time.sleep(0.3)
        page += 1
        time.sleep(1)
    return results

def get_yahoo_news(ticker, num_articles=20):
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        if not news_data:
            return []
        results = []
        for item in news_data[:num_articles]:
            title = item.get("content", {}).get("title", "")
            link = item.get("content", {}).get("canonicalUrl", {}).get("url", "")
            summary = item.get("content", {}).get("summary", "")
            if title:
                results.append({"title": title, "link": link, "body": summary})
        return results
    except:
        return []

def get_news(stock_info, num_articles=20):
    if stock_info["type"] == "KR":
        return get_naver_finance_news(stock_info, num_articles)
    else:
        return get_yahoo_news(stock_info["code"], num_articles)

# ========== ChromaDB 저장 ==========
def store_to_chromadb(news_list, stock_name, stock_code):
    docs = [
        Document(
            page_content=f"{news['title']}\n{news['body']}",
            metadata={"title": news["title"], "link": news["link"]}
        )
        for news in news_list if news["body"]
    ]
    if not docs:
        return None
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=f"./chroma_db/{stock_code}",
        collection_name=f"news_{stock_code}"
    )
    return vectorstore

# ========== RAG 체인 ==========
def build_rag_chain(vectorstore):
    llm = OllamaLLM(model="llama3.2", temperature=0.1)
    prompt_template = """
당신은 주식 뉴스 분석 전문가입니다.
아래 뉴스 기사들을 바탕으로 질문에 답해주세요.
반드시 한국어로 답변하고, 핵심 이슈를 간결하게 정리해주세요.

[참고 뉴스]
{context}

[질문]
{question}

[분석 결과]
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# ========== Streamlit UI ==========
st.set_page_config(page_title="📈 뉴스 기반 종목 이슈 탐색기", layout="wide")
st.title("📈 뉴스 기반 종목 이슈 탐색기")
st.caption("종목명을 입력하면 최신 뉴스를 분석해 AI가 이슈를 요약해드려요!")

# ✅ form으로 감싸서 엔터/버튼 둘 다 작동
with st.form("search_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_input = st.text_input("🔍 종목명 입력", placeholder="예: 삼성전자 / 엔비디아 / TSLA / 005930")
    with col2:
        num_articles = st.slider("수집 기사 수", min_value=5, max_value=30, value=20)
    submitted = st.form_submit_button("🚀 분석 시작")

if submitted:
    if not stock_input:
        st.warning("종목명을 입력해주세요!")
    else:
        with st.spinner("🔍 종목 검색 중..."):
            result = search_stock(stock_input.strip())
        
        if not result:
            st.error(f"'{stock_input}' 종목을 찾을 수 없어요. 영문명이나 티커로 다시 시도해보세요!")
        else:
            st.success(f"✅ {'[국내]' if result['type'] == 'KR' else '[해외]'} {result['name']} → {result['code']}")
            
            with st.spinner("📰 뉴스 수집 중..."):
                news = get_news(result, num_articles)
            
            if not news:
                st.error("뉴스를 가져오지 못했어요!")
            else:
                st.info(f"📄 총 {len(news)}개 기사 수집 완료!")
                
                with st.spinner("🗄️ 벡터 저장 중..."):
                    vectorstore = store_to_chromadb(news, result["name"], result["code"])
                
                if not vectorstore:
                    st.error("본문이 있는 기사가 없어요!")
                else:
                    with st.spinner("🤖 AI 분석 중..."):
                        chain = build_rag_chain(vectorstore)
                        query = f"{result['name']} 최근 주요 이슈는?"
                        answer = chain.invoke({"query": query})
                    
                    st.subheader("📊 AI 분석 결과")
                    st.write(answer["result"])
                    
                    with st.expander("📰 수집된 기사 목록 보기"):
                        for i, n in enumerate(news):
                            st.markdown(f"**{i+1}. {n['title']}**")
                            st.caption(n["link"])