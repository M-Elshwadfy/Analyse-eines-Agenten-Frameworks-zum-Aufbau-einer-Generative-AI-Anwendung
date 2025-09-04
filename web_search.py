from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from ddgs import DDGS
import logfire

logfire.configure()  
logfire.instrument_pydantic_ai()

provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("gpt-oss", provider=provider)

agent = Agent(
    model=model,
    instructions="You are a web search assistant. Use the web_search tool to fetch and summarize internet results."
)

# Diese Funktion durchsucht das Web mithilfe von DuckDuckGo (DDGS).
@agent.tool_plain
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with a given query"""
    # query: Suchbegriff(e), max_results: Anzahl gewünschter Treffer
    try:
        # DDGS-Session öffnen und Textsuche ausführen
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=max_results))
        # Treffer zusammenfassen: Titel, Snippet und Link
        summary = "\n".join([f"- {r['title']}: {r['body']} ({r['href']})" for r in results])
        # Formatiertes Ergebnis zurückgeben oder „No results.“
        return f"Results for '{query}':\n{summary}" if summary else "No results."
    except Exception as e:
        return f"Error: {str(e)}" # Fehlerfall (z. B. Netzwerk/Ratelimit) als String zurückgeben

user_prompt = "Search the web for: What is Pydantic AI"
result = agent.run_sync(user_prompt)
print(result.output)
