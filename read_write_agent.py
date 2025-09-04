from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire

logfire.configure()  
logfire.instrument_pydantic_ai()

provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("qwen3:8b", provider=provider)

def generate_file_txt(ctx: RunContext[str], text: str) -> str:
    """Write text file to the exact file path provided via deps."""
    with open(ctx.deps, 'w', encoding='utf-8') as f:
        f.write(text)
    return f"File generated successfully at path: {ctx.deps}"

def read_txt_file(ctx: RunContext[str]) -> str:
    """Read content from a .txt file specified via deps."""
    with open(ctx.deps, 'r', encoding='utf-8') as f:
        return f.read()
    
toolset = FunctionToolset(tools=[generate_file_txt, read_txt_file])

agent = Agent(
    model=model,
    deps_type=str,                 
    toolsets=[toolset],
    instructions=(

    "When asked to save text, call generate_file_txt. The file path comes from deps."
    "When asked to read text, call read_txt_file"
    )
)

result = agent.run_sync(
    "Save to text File: Meeting at 20, bring slides.",
    deps="/home/student/myenv/Text_Generator_Verzeichnis/Meeting.txt",
)
print(result.output)

result1 = agent.run_sync(
    "What is the content of the text file?",
    deps="/home/student/myenv/Text_Generator_Verzeichnis/Meeting.txt",
    )

print(result1.output)
