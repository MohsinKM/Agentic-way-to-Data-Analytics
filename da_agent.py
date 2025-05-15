import os
from dotenv import load_dotenv
import pandas as pd


# 1. Load your .env and API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env")

# 2. (Optional) Streaming callback to print tokens as they arrive
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager


from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# Use streaming=True + callback to see tokens/thought in real time
llm = ChatOpenAI(
    model="gpt-4-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    streaming=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

# 3. Read your DataFrame
df = pd.read_csv("../diabetes.csv")

# 4. Build the agent, capturing intermediate steps
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,                   # prints Thought/Action/Observation to stdout
    allow_dangerous_code=True,      # needed for Python execution
    return_intermediate_steps=True, # capture those steps in the return value
)

# 5. Run the agent and get both thoughts and final answer
user_question = "Show me the distribution of Glucose levels for positive and negative outcomes.?"
result = agent({"input": user_question})

# 6. Print out the agent's thought process:
print("\n--- Agent Thought Process ---\n")
for action, observation in result["intermediate_steps"]:
    # action.log contains the "Thought" text
    print(f"Thought: {action.log}")
    print(f"Action Input: {action.tool_input}")
    print(f"Observation: {observation}\n")

# 7. Finally, show the answer
print("--- Final Answer ---")
print(result["output"])


"""
Questions to ask: 
How many rows are there?
How many patients are in the dataset, and what percentage tested positive for diabetes?
Which columns contain zeros or missing values that may need imputation, and how many such entries are there
Show me the distribution of Glucose levels for positive and negative outcomes.
"""