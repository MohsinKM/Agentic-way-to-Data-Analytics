import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

# LangChain / OpenAI imports
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# 1) Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in your .env file and restart.")
    st.stop()

# 2) Streamlit page config
st.set_page_config(page_title="CSV RAG Explorer", page_icon="📊")
st.title("🔍 CSV RAG Explorer")

# 3) File uploader (only CSV)
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(
            f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns."
        )
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    # 4) Instantiate and cache the Pandas agent
    if "agent" not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        )
        st.session_state.agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            allow_dangerous_code=True,
        )

    # 5) Chat UI
    question = st.text_input("Ask a question about the CSV data:")
    if question:
        if st.button("Submit"):
            with st.spinner("Computing answer..."):
                try:
                    answer = st.session_state.agent.run(question)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Agent error: {e}")
else:
    st.info("👈 Please upload a CSV file to begin.")
