import os
from dotenv import load_dotenv
import pandas as pd

# Load your .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env")

# Imports from the new packages
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# ead your CSV
df = pd.read_csv("diabetes.csv")

# Instantiate the LLM client
llm = ChatOpenAI(
    model="gpt-4-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)

# Build the Pandas DataFrame agent
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True      # required for Python execution
)

# Ask your question
response = agent.run("How many rows are there?")
print(response)
