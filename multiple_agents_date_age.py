from datetime import datetime
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire

# Konfiguration des Loggings zur Analyse der Kommunikation zwischen Agent, Modell und Tool
logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

system_prompt = "Be Concise" # System-Prompt für das Verhalten des LLMs festlegen
# Verbindung zum lokalen Ollama-Server aufbauen
provider = OpenAIProvider(base_url="http://localhost:11434/v1")
# Modell „Qwen3:4B“ über den Provider initialisieren
model_1 = OpenAIModel("qwen3:4b", provider=provider)
# Agent mit Modell, System-Prompt, Tool-Definition und Wiederholungsanzahl erstellen
agent_1 = Agent(
    model=model_1,
    deps_type=int,  # Erwarteter Datentyp für die Abhängigkeit (hier: int)
    system_prompt=system_prompt,  # Verhaltensregel für das Modell
    instructions="Call tool get_user_birth_year, Greet User with his name and birth year",  # Anleitung für Tool-Nutzung
    retries=3  # Anzahl erlaubter Wiederholungen bei Fehlern
)
@agent_1.tool # Tool-Funktion definieren, das den Geburtsjahrwert zurückgibt
def get_user_birth_year(ctx: RunContext[int]) -> int:
    """Returns birth year of User using dependency"""
    return ctx.deps
# Agent ausführen mit Nutzereingabe und übergebener Jahreszahl
user_1_prompt = "Hi I am John"
result_1 = agent_1.run_sync(user_1_prompt, deps=1998)
print("First Agent Output:", result_1.output) # Ergebnis anzeigen

######################################################################

# Zweites Modell (Qwen2.5:7B) initialisieren
model_2 = OpenAIModel("qwen2.5:7b", provider=provider)
# Agent mit Anweisung zur Altersberechnung und Begrüßung erstellen
agent_2 = Agent(
    model=model_2,
    instructions=(
        "if user asks about his age Call get_current_year and figure out his age from current year and birth year"
        "Greet user with name and age"
    )
)
# Tool-Funktion, die das aktuelle Jahr zurückgibt
@agent_2.tool_plain
def get_current_year() -> int:
    """Returns actuall year"""
    return "Current year is", datetime.now().year

# Agent ausführen und dabei Nachrichtenverlauf vom ersten Agenten übernehmen
user_2_prompt = "How old am I?"
result_2 = agent_2.run_sync(user_2_prompt, message_history=result_1.new_messages())
print("Second Agent Output:", result_2.output) # Ergebnis anzeigen

###########################################################################################################

# Drittes Modell (Qwen3:1.7B) initialisieren
model_3 = OpenAIModel("qwen3:1.7b", provider=provider)
# Einfacher Agent ohne Tools erstellen
agent_3 = Agent(model=model_3)
# Nutzereingabe zur historischen Ereignisabfrage
user_3_prompt = "Tell me one big event happened in my birth year?"
# Agent ausführen und gesamten bisherigen Nachrichtenverlauf übergeben
result_3 = agent_3.run_sync(user_3_prompt, message_history=result_2.all_messages())
print("Third Agent Output:", result_3.output) # Ergebnis anzeigen




