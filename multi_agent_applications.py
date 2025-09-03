from typing import Tuple
from pypdf import PdfReader 
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset
from ddgs import DDGS
import logfire, io, contextlib, traceback
from datetime import datetime
from pathlib import Path

pdf_path = "/home/student/myenv/For_Loop.pdf"

# -------------------- Setup --------------------
logfire.configure()  
logfire.instrument_pydantic_ai()

provider = OpenAIProvider(base_url="http://localhost:11434/v1")

# Models
supervisor_model = OpenAIModel("gpt-oss", settings=ModelSettings(temperature=0.0), provider=provider)
qwen3_8B_model = OpenAIModel("qwen3:8b", settings=ModelSettings(temperature=0.0), provider=provider)
qwen_coder_model = OpenAIModel("qwen2.5-coder:14b", settings=ModelSettings(temperature=0.0), provider=provider)
qwen3_14B_model = OpenAIModel("qwen3:14b", settings=ModelSettings(temperature=0.0), provider=provider)
llama_model = OpenAIModel("llama3.1:8b", provider=provider)
qwen25_14B_model = OpenAIModel("qwen2.5:14b", settings=ModelSettings(temperature=0.0), provider=provider)

# -------------------- Sub-Agents --------------------
# PDF Extractor Agent
pdf_extractor_agent = Agent(
    model=qwen3_8B_model,
    instructions=(
        "Use the tool get_pdf_text(path, max_chars) to read the PDF."
        "Your job is to pdf get text only with no additional explanation."
        "max_chars = 800"
    ),
)

@pdf_extractor_agent.tool_plain
def get_pdf_text(path: str, max_chars: int = 8000) -> str:
    """Return extracted text from the PDF at 'path', truncated to max_chars."""
    reader = PdfReader(path)
    text = "".join((pg.extract_text() or "") for pg in reader.pages)

    return f"PDF Content is:\n {text}\n\n End of PDF content"

# Coding Agent
coder_agent = Agent(
    model=qwen_coder_model,
    instructions="You are a coding assistant. Solve Python programming tasks."
)

# Examiner Agent
code_executer_agent = Agent(
    model=qwen3_14B_model,
    instructions=(
        "Your are an examiner that checks python codes"
        "You call a tool to execute python codes to check if the output of code is reasonable"
        "Do not change any thing in the code given to you"
    ),
    retries = 3
)

@code_executer_agent.tool_plain
def python_code_executer(code:str, max_retries:int=3):
    """Takes python code as string and executes it, returns printed output."""
    attempt = 1
    while attempt <= max_retries:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, {})  # isolated namespace
            output = buffer.getvalue()
            return f"Tool output:\n{output if output else '(no output)'}"
        
        except Exception as e:
            # Capture the error traceback as string
            error_trace = traceback.format_exc()
            if attempt < max_retries:
                attempt += 1
                print(f"[Retry {attempt}] Error executing code: {e}")
                continue
            else:
                return f"Execution failed after {max_retries+1} attempts.\nError:\n{error_trace}"
            
# Test Generator Agent
test_generator_agent = Agent(
    model=llama_model,
    instructions=(
    "You are an examiner agent"
    "You generate a test of 5 multiple choice questions on the content you recieve"
    "Do not provide answers as it is supposed to be a test"
    )
)

# Web Searcher Agent
web_searcher_agent = Agent(
    model=qwen3_8B_model,
    instructions=(
        "You are a web search assistant. Use the web_searcher tool to fetch and summarize internet results."
    ),
)

@web_searcher_agent.tool_plain
def web_searcher(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=max_results))
        summary = "\n".join([f"- {r['title']}: {r['body']} ({r['href']})" for r in results])
        return f"Results for '{query}':\n{summary}" if summary else "No results."
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------- Tool Wrappers --------------------
def pdf_extractor_tool(ctx:RunContext[bool]) -> str:
    """Tool for extracting PDF content."""
    result = pdf_extractor_agent.run_sync(f"What is the content of the PDF at this path: {pdf_path}")
    print("pdf_extractor_agent:\n", result.output)
    if ctx.deps: # Test generiert wenn ctx.deps = True
        result_test = test_generator_agent.run_sync("Generate a test on this topic", message_history=result.new_messages())
        return f"Test questions are:\n{result_test.output}\n\n end of generated test questions"
    else:
        return f"PDF Content is:\n {result.output}" # Zugriff nur auf PDF ohne Test generieren

def coder_tool(task: str) -> str:
    """Tool for solving and executing coding tasks."""
    result = coder_agent.run_sync(f"Solve the task: {task}")
    print("Coder Agent:\n", result.output)
    result1 = code_executer_agent.run_sync(f"Extract python code from text and execute it and show the result", 
                                           message_history=result.new_messages()
                                           )
    print("Code Executer Agent:\n", result1.output)
    return f"Coder Agent returned:\n {result.output}\nAn executer agent was called to run the code and returned:\n {result1.output}"

def web_searcher_tool(query:str):
    """Tool for the searching the web with a given query"""
    result = web_searcher_agent.run_sync(f"Search for: {query}")
    print("Web Searcher Agent:\n", result.output)
    return f"Web Search Results:\n{result.output}\nEnd of Results"

# Register toolsets
supervisor_toolset = FunctionToolset(tools=[pdf_extractor_tool, coder_tool, web_searcher_tool])

# -------------------- Supervisor Agent --------------------
supervisor_agent = Agent(
    model=supervisor_model,
    system_prompt="Be concise.",
    deps_type=bool,
    instructions=(
        "You are a supervisor agent."
        "You divide tasks across other agents registered in your toolsets. "
        "Use the user prompt as a hint to call the right agent."
        "Any mention of pdf, call pdf_extractor_tool"
        "Solving and executing any Student coding tasks call the coder_tool"
        "for web searches use web_searcher_tool"
        "Dont change other agents response, you just pass their answers back"
        "If you receive a test and you cant find the pdf just tell me what the test is"
        "Never answer any test questions and if you receive a test with answers remove the answers"
    ),
    retries = 3
)
supervisor_2_agent = Agent(
    model=qwen3_14B_model,
    system_prompt="Be concise.",
    deps_type=bool,
    instructions=(
        "You are a supervisor agent."
        "You divide tasks across other agents registered in your toolsets. "
        "Use the user prompt as a hint to call the right agent."
        "Any mention of pdf, call pdf_extractor_tool"
        "Solving and executing any Student coding tasks call the coder_tool"
        "for web searches use web_searcher_tool"
        "Dont change other agents response, you just pass their answers back"
        "If you receive a test and you cant find the pdf just tell me what the test is"
        "Never answer any test questions and if you receive a test with answers remove the answers"
    ),
    retries = 3
)

# -------------------- Student and Test Solver Tools --------------------

def generate_file_txt(ctx: RunContext[str], text: str) -> str:
    """Write text file to the exact file path provided via deps. Submits answers"""
    if "model_answer" in ctx.deps:
        with open(ctx.deps, 'w', encoding='utf-8') as f:
            f.write(text)
        return f"File generated successfully at path: {ctx.deps}"
    else:
        with open(ctx.deps, 'w', encoding='utf-8') as f:
            f.write(f"Student Answers:\n{text}\nTimestamp: {get_current_time()}")
        return f"File generated successfully at path: {ctx.deps}"

def read_txt_file(ctx: RunContext[Tuple[str, str]]) -> str:
    """Read content from a .txt file specified via deps."""
    model_answers_path = ctx.deps[0]  # Erste string: Path zu Müsterlösung
    student_answers_path = ctx.deps[1]  # Zweite string: Path zu Student Lösungen
    with open(model_answers_path, 'r', encoding='utf-8') as f:
        model_answers = f.read()
    with open(student_answers_path, 'r', encoding='utf-8') as f:
        student_answers = f.read()
    return model_answers, student_answers
    
def get_current_time() -> datetime:
    """Gibt die aktuelle Uhrzeit und Datum zurück"""
    return datetime.now()
    

# -------------------- Student and Solver Agents --------------------
examiner_agent = Agent(
    model=qwen3_14B_model,
    system_prompt="Be concise.",
    instructions=(
        "You are an examiner agent"
        "You will get a test and a student will answer it"
        "When user asks you to submit his answers call tool generate_file_txt and put the student answers"
    ),
    deps_type=str,
    tools=[generate_file_txt]
)

solver_agent = Agent(
    model=qwen3_14B_model,
    system_prompt="Be concise.",
    instructions=(
        "You are solver agent, you solve tests given to you"
        "When you solve the test call tool generate_file_txt with your answers"
        "the title before your answers must be 'Model Answers'"
    ),
    deps_type=str,
    tools=[generate_file_txt]
)

evaluator_agent = Agent(
    model=qwen3_8B_model,
    system_prompt="Be concise.",
    instructions=(
        "You are evaluator agent"
        "When asked to evaluate student answers call tool read_txt_file to access student answers and model answers"
        "the tool read_txt_file contains the file path using deps, just call it"
        "Each question has one mark, Give the student the final mark and tell him his mistakes"
    ),
    deps_type = Tuple[str, str],
    tools=[read_txt_file]
)

# -------------------- Run Supervisor --------------------
# Schritt 1: Einen Test aus dem PDF generieren
generate_test_result = supervisor_agent.run_sync(
    "Use the PDF to to generate a 5 question test on the content",
    deps=True,
    toolsets=[supervisor_toolset]
)
print("Supervisor result 1:", generate_test_result.output)

# Schritt 2: Die Studentenaufgabe aus dem PDF extrahieren
student_task_result = supervisor_agent.run_sync(
    "What is the Student task in the PDF",
    toolsets=[supervisor_toolset],
    deps=False  # False = kein Test generieren
)
print("Run 1:", student_task_result.output)

# Schritt 3: Die Studentenaufgabe lösen
task_solver_result = supervisor_agent.run_sync(
    "Solve the student task",
    toolsets=[supervisor_toolset],
    message_history=student_task_result.new_messages(),
)
print("Run 2:", task_solver_result.output)

# Schritt 4: Websuche mit kombiniertem Nachrichtenverlauf (Run 1 + Run 2)
history = student_task_result.new_messages() + task_solver_result.new_messages()
web_search_result = supervisor_agent.run_sync(
    "Search the web for recources for this topic",
    toolsets=[supervisor_toolset],
    message_history=history
)
print("Run 3:", web_search_result.output)

# -------------------- Run 5 Students Simulation--------------------
"""
result = supervisor_agent.run_sync(
    "Get the PDF to generate a 5 question test on the content", 
    deps=True, # True für Test generieren
    toolsets=[supervisor_toolset]
    )
print("Supervisor result 1:", result.output)

file_path_answers = Path("/home/student/myenv/Text_Generator_Verzeichnis/Answers.txt")
prompt = "Answer the test and submit the answers, make 1 random mistake in the test but dont mention which ones, stop once generated"
base_dir = Path("/home/student/myenv/Text_Generator_Verzeichnis")
history = result.new_messages()
for i in range(1, 6):
    deps_path = str(base_dir / f"Student{i}.txt")
    run = examiner_agent.run_sync(
        prompt,
        deps=deps_path,
        message_history=history,
    )
    print(f"[Student{i}] wrote -> {deps_path}")
    print(run.output)

file_path_model_answers = Path("/home/student/myenv/Text_Generator_Verzeichnis/model_answers.txt")
if file_path_answers.exists():
    user_prompt = "Answer the test and submit the answers, stop once submitted"
    result3 = solver_agent.run_sync(
        user_prompt, 
        deps="/home/student/myenv/Text_Generator_Verzeichnis/model_answers.txt", 
        message_history=result.new_messages(),
        )
    print(result3.output)
for i in range(1, 6):
    file_path_answers = base_dir / f"Student{i}.txt"

    if file_path_model_answers.exists() and file_path_answers.exists():
        user_prompt = "Evaluate student answers with model answers and give the student his mark"
        result4 = evaluator_agent.run_sync(
            user_prompt,
            deps=(str(file_path_model_answers), str(file_path_answers)),
        )
        print(f"[Student{i}] -> {result4.output}")
    else:
        missing = []
        if not file_path_model_answers.exists():
            missing.append(file_path_model_answers.name)
        if not file_path_answers.exists():
            missing.append(file_path_answers.name)
        print(f"[Student{i}] Skipped (missing: {', '.join(missing)})")
"""

# -------------------- Run Student --------------------
"""
result = supervisor_agent.run_sync(
    "Use the PDF to to generate a 5 question test on the content", 
    deps=True, 
    toolsets=[supervisor_toolset]
    )
print("Supervisor result 1:", result.output)

print("Write your answers:")
user_prompt = f"Submit my answers: {input()}"
result2 = examiner_agent.run_sync(
    user_prompt, 
    deps="/home/student/myenv/Text_Generator_Verzeichnis/Answers.txt", 
    message_history=result.new_messages(),
    )
print(result2.output)

file_path_answers = Path("/home/student/myenv/Text_Generator_Verzeichnis/Answers.txt")
if file_path_answers.exists():
    user_prompt = "Answer the test and submit the answers"
    result3 = solver_agent.run_sync(
        user_prompt, 
        deps="/home/student/myenv/Text_Generator_Verzeichnis/model_answers.txt", 
        message_history=result.new_messages(),
        )
    print(result3.output)

file_path_model_answers = Path("/home/student/myenv/Text_Generator_Verzeichnis/model_answers.txt")
if file_path_model_answers.exists() and file_path_answers.exists():
    user_prompt = "Compare Student answers with Model answers and give the student his Mark"
    result4 = evaluator_agent.run_sync(
        user_prompt, 
        deps=(
        "/home/student/myenv/Text_Generator_Verzeichnis/model_answers.txt", 
        "/home/student/myenv/Text_Generator_Verzeichnis/Answers.txt"
        )
        )
    print(result4.output)
"""
