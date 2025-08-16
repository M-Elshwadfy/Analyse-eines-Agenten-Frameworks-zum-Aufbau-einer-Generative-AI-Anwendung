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
model = OpenAIModel("qwen2.5:7b", provider=provider)

# Initialisierung eines Agenten, der über das Modell kommuniziert
# Der Agent übernimmt die Kommunikation mit dem Modell, orchestriert Tool-Calls und verarbeitet Antworten
system_prompt = (
    "Falls nach der Uhrzeit und Datum gefragt wird, verwende die Funktion get_current_time(), "
    "um die aktuelle Zeit und Datum zu ermitteln. "
    "Antwort in diese Fromat: Es ist (Uhrzeit) Uhr am Tag. Monat in Buchstaben, Jahr"
)
agent = Agent(model=model, system_prompt=system_prompt)
# Tool definieren, Durch den Dekorator @agent.tool_plain wird die Funktion als "nutzbar für das Modell" registriert.
@agent.tool_plain
def get_current_time() -> datetime:
    """Gibt die aktuelle Uhrzeit und Datum zurück"""
    datum = datetime.now().strftime("%d-%m-%Y")
    uhrzeit = datetime.now().strftime("%H:%M:%S")
    return f"Uhrzeit ist {uhrzeit}, datum ist {datum}"

# Agent mit Benutzereingabe aufrufen
result = agent.run_sync('Was ist die aktuelle Datum und Uhrzeit?')
date_now = get_current_time()
print(date_now)
print(result.output)
