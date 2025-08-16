from datetime import datetime
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire  # type: ignore

# Konfiguration des Loggings zur Analyse der Kommunikation zwischen Agent, Modell und Tool
logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("llama3.1:8b", provider=provider)

agent = Agent(
    model=model,    
    deps_type=int, # Übergabe eines externen Wertes (Geburtsjahr) an den Agenten über „deps“
    system_prompt=(
        "Birth year of user is known with Tool get_player_birth_year"
        "Call get_current_time from tools to get the current year and then subtract birth year from it to calculate the age"
        #"Frage den Nutzer nicht nach seinem Geburtsjahr."
    ),
    retries=3
)

@agent.tool_plain
def get_current_time() -> datetime:
    """Returns actuall time"""
    return datetime.now()
@agent.tool
def get_player_birth_year(ctx: RunContext[int]) -> int:
    """Returns birth year of User using dependency"""
    return ctx.deps
# Erste Anfrage: Alter berechnen auf Basis des Geburtsjahres
user_prompt = "How old am I?"
result1 = agent.run_sync(user_prompt, deps=1998)
print(result1.output)
# Zweite Anfrage mit Nachrichtenverlauf aus der ersten Interaktion
# Übergibt den vorherigen Gesprächskontext an das Modell, um konsistente Antworten zu ermöglichen
result2 = agent.run_sync("Who won the world cup on my year of brith?", message_history=result1.new_messages())
print(result2.output)

# Public Link for Logfire for this code
# https://logfire-eu.pydantic.dev/shared-trace/45c20eac-86da-4171-85a9-308057f3a385