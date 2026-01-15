from crewai import Agent
from tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os


## call the gemini models
# Model name can be configured via GEMINI_MODEL env variable
# Defaults to gemini-pro (most widely available)
gemini_model = os.getenv("GEMINI_MODEL", "gemini-pro")
llm=ChatGoogleGenerativeAI(model=gemini_model,
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

# Creating a senior researcher agent with verbose mode
# Note: memory parameter removed as it's not a boolean in latest CrewAI

article_researcher=Agent(
    role="Senior Researcher",
    goal='Uncover ground breaking technologies in {topic}',
    verbose=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of "
        "innovation, eager to explore and share knowledge that could change "
        "the world."
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=True
)

## creating a write agent with custom tools responsible in writing news blog

article_writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft "
    "engaging narratives that captivate and educate, bringing new "
    "discoveries to light in an accessible manner."
  ),
  tools=[tool],
  llm=llm,
  allow_delegation=False
)

