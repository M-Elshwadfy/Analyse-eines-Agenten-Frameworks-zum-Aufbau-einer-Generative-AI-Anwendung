from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire

# Konfiguration des Loggings zur Analyse der Kommunikation zwischen Agent, Modell und Tool
logfire.configure()
logfire.instrument_pydantic_ai()

# Ollama-Provider und Modell konfigurieren
provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("qwen3:8b", provider=provider)

# Agent mit Modell initialisieren
agent = Agent(model=model, instructions="Use Tool evaluate_expression() for calculations")

@agent.tool_plain
def evaluate_expression(expression: str) -> float:
    """Evaluate a basic math expression"""
    allowed_names = {"__builtins__": {}}  # Zugriff auf eingebaute Funktionen sperren
     # eval() wertet den String-Ausdruck aus und liefert das numerische Ergebnis
    return eval(expression, allowed_names) 

# Beispielanfrage an den Agenten
user_prompt = "What is 545.38*74.62/6.83?"
result = agent.run_sync(user_prompt)
print(result.output)
