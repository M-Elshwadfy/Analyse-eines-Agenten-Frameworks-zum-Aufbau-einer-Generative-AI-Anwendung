from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire

# Konfiguration des Loggings zur Analyse der Kommunikation zwischen Agent, Modell und Tool
logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

# Ollama-Provider und Modell konfigurieren
provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("granite3.3:8b", provider=provider)

# Agent mit Modell initialisieren
agent = Agent(model=model)

@agent.tool_plain
def evaluate_expression(expression: str) -> float:
    """Evaluate a basic math expression"""
    allowed_names = {"__builtins__": {}}  # Zugriff auf eingebaute Funktionen sperren
    return eval(expression, allowed_names)

# Beispielanfrage an den Agenten
response = agent.run_sync("Calculate 8*9*2*3/1.34")
print(response.output)
