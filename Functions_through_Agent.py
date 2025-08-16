from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire # type: ignore

# Konfiguration des Loggings zur Analyse der Kommunikation zwischen Agent, Modell und Tool
logfire.configure()  
logfire.instrument_pydantic_ai()

def get_current_time() -> datetime:
    """Gibt die aktuelle Uhrzeit zurück"""
    return datetime.now()

def add_numbers(a: float, b: float) -> float:
    """Gibt die Summe zweier Zahlen zurück"""
    return a + b

# Definieren des LLM-Providers
provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("llama3.1:8b", provider=provider)

# Initialisierung eines Agenten, der über das Modell kommuniziert
# Der Agent übernimmt die Kommunikation mit dem Modell, orchestriert Tool-Calls und verarbeitet Antworten
agent = Agent(model=model, tools=[get_current_time, add_numbers])

# Agent mit Benutzereingabe aufrufen
result = agent.run_sync('Wie spät ist es?') 
print(result.output)

