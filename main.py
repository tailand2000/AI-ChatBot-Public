import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os 
from dotenv import load_dotenv

load_dotenv()

# LangSmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたはヘルプアシスタントです。 ユーザーの質問に回答してください。"),
        ("user", "質問:{question}"),
    ]
)

def generate_response(question, model, temperature, max_tokens):
    openai.api_key = os.environ.get("OPENAI_AIPI_KEY")
    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

with st.sidebar:
    st.title("Setting LLM")
    # apikey = st.text_input("あなたのOpenAIのAPIキーを入力してください。", type="password")
    # openAIモデルのドロップダウンリスト
    model = st.selectbox("モデル選択", ["gpt-4o-mini", "gpt-4o", "gpt-4o-turbo"])

    # パラメータの調整
    temperature = st.slider("Tempatature", min_value=0.0, max_value=1.0, value=0.7)
    max_tokens = st.slider("Max Token", min_value=100, max_value=500, value=100)

st.title("OpenAI 質問チャットボット")
st.write("あなたの質問にAIが回答してくれます！")
user_input = st.text_input("質問を入力してください。")

if st.button("送信", key="submit"):
    if user_input:
        response = generate_response(user_input, model, temperature, max_tokens)
        st.write(response)
    else:
        st.warning("質問を入力してください。")

