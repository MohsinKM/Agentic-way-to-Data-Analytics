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
    # return_intermediate_steps=True, # capture those steps in the return value
)

report_prompt = """
1. Load the file "diabetes.csv" into a Pandas DataFrame.
2. Clean the data:
   a. In columns SkinThickness and Insulin, replace zeros with the column median.
   b. Report how many replacements you made.
3. Perform exploratory analysis:
   a. Histogram of Glucose, faceted by Outcome.
   b. Boxplot of BMI grouped by Outcome.
   c. Correlation heatmap of all numeric features.
   d. Any other 1–2 plots you think are interesting.
4. For each plot, write a 1–2 sentence insight underneath.
5. Use matplotlib (or seaborn) and the PdfPages class from matplotlib.backends to assemble **all** your figures into a single PDF file named "diabetes_report.pdf".
6. Save that PDF to disk, and at the end print out the filename and a short summary of the key takeaways.
"""

result = agent.run(report_prompt)
print("Done – created:", result)


