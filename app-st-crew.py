#https://docs.crewai.com/how-to/Installing-CrewAI/
#https://docs.crewai.com/how-to/LLM-Connections/#ollama-integration-ex-for-using-llama-2-locally

# OPENAI_API_BASE='http://localhost:11434/v1'
# OPENAI_MODEL_NAME='llama3:latest'  # Adjust based on available model
# OPENAI_API_KEY='ollama'

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    api_key="ollama",
    model="llama3:latest",
    #model="splitpierre/bode-alpaca-pt-br:latest", 
    #model="splitpierre/bode-alpaca-pt-br:13b-Q4_0", 
    base_url="http://localhost:11434/v1",
)

general_agent = Agent(role = "Math Professor",
                      goal = """Provide the solution to the students that are asking mathematical questions and give them the answer.""",
                      backstory = """You are an excellent math professor that likes to solve math questions in a way that everyone can understand your solution""",
                      allow_delegation = False,
                      verbose = True,
                      llm = llm)

task = Task(description="""what is 3 + 5""",
             agent = general_agent,
             expected_output="A numerical answer.")

crew = Crew(
            agents=[general_agent],
            tasks=[task],
            verbose=2
        )

result = crew.kickoff()

print(result)