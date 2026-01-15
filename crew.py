from crewai import Crew, Process
from tasks import research_task, write_task
from agents import article_researcher, article_writer, llm

## Forming the tech focused crew with some enhanced configuration
# Explicitly set the LLM for the crew to use Gemini
crew = Crew(
    agents=[article_researcher, article_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    llm=llm  # Use Gemini LLM for the crew
)

## starting the task execution process wiht enhanced feedback

result=crew.kickoff(inputs={'topic':'tell me about food'})
print(result)