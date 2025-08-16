from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire

# Aktiviert das Log-System für die Analyse der Agentenkommunikation
logfire.configure()  
logfire.instrument_pydantic_ai()  # Überwacht alle Vorgänge innerhalb von PydanticAI
logfire.instrument_httpx(capture_all=True)  # Protokolliert alle HTTP-Anfragen an den LLM-Server

# Verbindet sich mit dem lokal laufenden Ollama-Server über Port 11434
provider = OpenAIProvider(base_url="http://localhost:11434/v1")
# Wählt das Modell „qwen2.5:7b“, das lokal über Ollama ausgeführt wird
model = OpenAIModel("qwen2.5:7b", provider=provider)
# Erstellt einen Agenten mit System Prompt, um Verhalten zu steuern
agent = Agent(model=model, system_prompt="Grüße den Benutzer mit Namen und nenne sein Alter.")
# Registriert eine einfache Funktion als Tool für den Agenten – berechnet das Alter aus Geburtsjahr
@agent.tool_plain
def calculate_age(name: str, birth_year: int) -> str:
    """Gibt Benutzer Name und Alter zurück"""
    current_year = datetime.now().year  # Holt das aktuelle Jahr
    age = current_year - birth_year  # Berechnet das Alter
    return f"{name} Alter ist {age}"

# Schickt eine natürliche Spracheingabe an den Agenten – LLM extrahiert Daten und ruft ggf. das Tool auf
user_prompt = "Mein Name ist Max und ich wurde 1995 geboren. Wie alt bin ich?"
result = agent.run_sync(user_prompt)
print(result.output)



print("")

