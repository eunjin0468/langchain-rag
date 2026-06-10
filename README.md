# langchain-rag

## 기술 스택

| 역할 | 기술 |
|------|------|
| LLM | Ollama + llama3.2 (로컬) |
| 임베딩 | HuggingFace BGE-m3 |
| 벡터 DB | ChromaDB |
| 프레임워크 | LangChain |
| 크롤링 | requests + BeautifulSoup |
| UI | Streamlit |

## 주요 기능

- 네이버 금융 뉴스 크롤링 (국내 종목)
- 야후 파이낸스 뉴스 수집 (해외 종목)
- ChromaDB 벡터 저장 및 유사도 검색
- llama3.2 기반 RAG 이슈 요약

## 실행 방법

```bash
# 가상환경 활성화
cd D:\tmp
venv\Scripts\activate

# Streamlit 실행
streamlit run app.py
```

## 구조
```
langchain-rag/
├── stock_news_rag.ipynb   
├── app.py           # Streamlit UI
└── chroma_db/       # 벡터 DB (로컬 저장)
```
