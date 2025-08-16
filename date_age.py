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
model = OpenAIModel("qwen3:8b", provider=provider)

agent = Agent(
    model=model,    
    deps_type=int, # Übergabe eines externen Wertes (Geburtsjahr) an den Agenten über „deps“
    system_prompt=(
        "Das Geburtsjahr des Nutzers wird über das Tool get_player_birth_year bereitgestellt. "
        "Rufe zuerst get_current_time auf, um das aktuelle Jahr zu erfahren, und subtrahiere dann das Geburtsjahr, um das Alter zu berechnen. "
        "Frage den Nutzer nicht nach seinem Geburtsjahr."
    ),
    retries=3
)

@agent.tool_plain
def get_current_time() -> datetime:
    """Gibt die aktuelle Uhrzeit zurück"""
    return datetime.now()
@agent.tool
def get_player_birth_year(ctx: RunContext[int]) -> int:
    """Gibt das Geburtsjahr zurück, das als Abhängigkeit (deps) übergeben wurde."""
    return ctx.deps
# Erste Anfrage: Alter berechnen auf Basis des Geburtsjahres
user_prompt = "Wie alt bin ich?"
result1 = agent.run_sync(user_prompt, deps=1998)
print(result1.output)
# Zweite Anfrage mit Nachrichtenverlauf aus der ersten Interaktion
# Übergibt den vorherigen Gesprächskontext an das Modell, um konsistente Antworten zu ermöglichen
result2 = agent.run_sync("Wer hat die Weltmeisterschaft in meinem Geburtsjahr gewonnen?", message_history=result1.new_messages())
print(result2.output)

#https://logfire-eu.pydantic.dev/mohamed-elshwadfy98/pydanticai?q=trace_id%3D%2701985462174f568c4bd0a411231d23a3%27+and+span_id%3D%27edcd5a37a86b558e%27&spanId=edcd5a37a86b558e&traceId=01985462174f568c4bd0a411231d23a3&env=-clear-&since=2025-07-29T04%3A12%3A51.407816Z&until=2025-07-29T04%3A13%3A44.721516Z