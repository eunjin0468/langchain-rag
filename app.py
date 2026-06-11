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
import plotly.graph_objects as go
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
                return body.text.strip()
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

def get_stock_chart(stock_info):
    try:
        if stock_info["type"] == "KR":
            ticker = stock_info["code"] + ".KS"
        else:
            ticker = stock_info["code"]
        
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if df.empty:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="주가"
        ))
        fig.update_layout(
            title=f"{stock_info['name']} 3개월 주가",
            xaxis_rangeslider_visible=False,
            height=400
        )
        return fig
    except:
        return None
    
# ========== ChromaDB 저장 ==========
def store_to_chromadb(news_list, stock_name, stock_code):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    docs = []
    for news in news_list:
        if news["body"]:
            chunks = splitter.split_text(news["body"])
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata={"title": news["title"], "link": news["link"]}
                ))
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
st.set_page_config(
    page_title="StockLens",
    layout="wide",
    initial_sidebar_state="collapsed"
)

css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero {
        padding-top: 48px;
        padding-bottom: 32px;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.6;
        font-weight: 400;
    }

    .result-card {
        border-radius: 16px;
        padding: 28px 32px;
        margin-top: 24px;
        border: 1px solid rgba(128,128,128,0.15);
        font-size: 0.97rem;
        line-height: 1.8;
    }

    .result-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #0066CC;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }

    .news-item {
        padding-top: 14px;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(128,128,128,0.15);
    }
    .news-item:last-child { border-bottom: none; }
    .news-title {
        font-size: 0.93rem;
        font-weight: 500;
    }
    .news-link {
        font-size: 0.8rem;
        color: #0066CC;
        text-decoration: none;
    }

    .badge {
        display: inline-block;
        background: rgba(0,102,204,0.12);
        color: #0066CC;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 16px;
    }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown("""
    <div class="hero">
        <h1>AI가 읽은 오늘의 종목, 한눈에 보기</h1>
        <p>궁금한 종목, 지금 바로 분석해보세요</p>
    </div>
""", unsafe_allow_html=True)

with st.form("search_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_input = st.text_input(
            "어떤 종목이 궁금하세요?",  
            placeholder="삼성전자, 엔비디아, TSLA, 005930 …",
        )
    with col2:
        num_articles = st.slider("기사 몇 개 볼까요?", min_value=5, max_value=30, value=20)
    query_input = st.text_input(
    "어떤 게 궁금하세요?",
    placeholder="최근 주요 이슈는? / 최근 실적은? / 리스크 요인은?",
    )
    submitted = st.form_submit_button("분석하기")

if submitted:
    if not stock_input:
        st.warning("종목명을 입력해주세요!")
    else:
        with st.spinner("종목 찾는 중 ..."):
            result = search_stock(stock_input.strip())

        if not result:
            st.error(f"'{stock_input}' 해당 종목을 찾지 못했어요. 다른 이름이나 티커로 시도해보세요.")
        else:
            market = "국내" if result["type"] == "KR" else "해외"
            st.markdown(f'<div class="badge">{market} · {result["code"]}</div>', unsafe_allow_html=True)

            with st.spinner("뉴스 긁어오는 중 ..."):
                news = get_news(result, num_articles)

            if not news:
                st.error("뉴스를 불러오지 못했어요. 잠시 후 다시 시도해주세요.")
            else:
                with st.spinner("AI가 읽는 중 ..."):
                    vectorstore = store_to_chromadb(news, result["name"], result["code"])

                if not vectorstore:
                    st.error("본문이 있는 기사가 없어요.")
                else:
                    with st.spinner("답변 만드는 중 ..."):
                        chain = build_rag_chain(vectorstore)
                        query = f"{result['name']} {query_input}"
                        answer = chain.invoke({"query": query})

                    col_chart, col_result = st.columns([1, 1])

                    with col_chart:
                        with st.spinner("차트 불러오는 중"):
                            fig = get_stock_chart(result)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("차트 데이터를 불러오지 못했어요.")
                    
                    with col_result:
                        st.markdown(f"""
                            <div class="result-card">
                                <div class="result-label">AI 분석</div>
                                {answer["result"].replace(chr(10), "<br>")}
                            </div>
                        """, unsafe_allow_html=True)

                    with st.expander(f"수집된 기사 {len(news)}건"):
                        for i, n in enumerate(news):
                            st.markdown(f"""
                                <div class="news-item">
                                    <div class="news-title">{i+1}. {n['title']}</div>
                                    <a class="news-link" href="{n['link']}" target="_blank">{n['link']}</a>
                                </div>
                            """, unsafe_allow_html=True)