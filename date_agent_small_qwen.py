from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire # type: ignore

# Konfiguration des Loggings zur Analyse der Kommunikation zwischen Agent, Modell und Tool
logfire.configure()  
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("qwen2.5:1.5b", provider=provider)

# Initialisierung eines Agenten, der über das Modell kommuniziert
# Der Agent übernimmt die Kommunikation mit dem Modell, orchestriert Tool-Calls und verarbeitet Antworten
system_prompt="If asked about time call get_current_time() function from tools"
agent = Agent(model=model, system_prompt=system_prompt)
# Tool definieren, Durch den Dekorator @agent.tool_plain wird die Funktion als "nutzbar für das Modell" registriert.
@agent.tool_plain
def get_current_time() -> datetime:
    """Returns actuall time"""
    return datetime.now()

# Agent mit Benutzereingabe aufrufen
result = agent.run_sync('What time is it?')
print(result.output)
