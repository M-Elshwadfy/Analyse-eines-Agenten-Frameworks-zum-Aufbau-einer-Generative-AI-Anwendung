from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire

# Konfiguration des Loggings zur Analyse der Kommunikation zwischen Agent, Modell und Tool
logfire.configure()  
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

# Lokalen Ollama-Provider einrichten und das Modell „Qwen3:4B“ verbinden
provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("qwen3:4b", provider=provider)

# Initialisierung eines Agenten, der über das Modell kommuniziert
# Der Agent übernimmt die Kommunikation mit dem Modell, orchestriert Tool-Calls und verarbeitet Antworten
system_prompt="Falls nach der Uhrzeit und Datum gefragt wird, verwende die Funktion get_current_time()"
agent = Agent(model=model, system_prompt=system_prompt)

# Tool definieren, Durch den Dekorator @agent.tool_plain wird die Funktion als "nutzbar für das Modell" registriert.
@agent.tool_plain
def get_current_time() -> datetime:
    """Gibt die aktuelle Uhrzeit und Datum zurück"""
    return datetime.now()

# Agent mit Benutzereingabe aufrufen
user_prompt = "Was ist die aktuelle Datum und Uhrzeit?"
result = agent.run_sync(user_prompt)
print(result.output)




print("")
#https://logfire-eu.pydantic.dev/shared-trace/63d0ea06-6caf-4a8a-9b89-5d9e99bd67e2