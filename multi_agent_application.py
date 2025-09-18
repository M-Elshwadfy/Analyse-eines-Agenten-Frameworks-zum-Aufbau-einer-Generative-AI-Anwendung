from typing import Tuple                  # Typ-Hinweise (z. B. Tupel)
from pypdf import PdfReader                # PDF-Leser (je nach Paket)
from pydantic_ai import Agent, RunContext  # Kernklassen: Agent + Laufkontext
from pydantic_ai.models.openai import OpenAIModel  # Modell-Wrapper
from pydantic_ai.providers.openai import OpenAIProvider  # Provider (z. B. Ollama)
from pydantic_ai.settings import ModelSettings  # Einstellungen (Temp., Tokens)
from pydantic_ai.toolsets import FunctionToolset # Sammlung/Registrierung von Tools
from ddgs import DDGS                    # DuckDuckGo-Suche
import logfire, io, contextlib, traceback  # Logging & Hilfsfunktionen
from datetime import datetime            # Datum/Zeit
from pathlib import Path                 # Pfad-Objekte
import random

pdf_path = "/home/student/myenv/For_Loop.pdf"   # Pfad zur Beispiel-PDF

# ---------- Setup ----------
logfire.configure()                      # Logfire initialisieren
logfire.instrument_pydantic_ai()         # PydanticAI-Events mitschneiden

provider = OpenAIProvider(base_url="http://localhost:11434/v1")  # Ollama-Endpoint

# ---------- Modelle ----------
supervisor_model = OpenAIModel(
    "gpt-oss",
    settings=ModelSettings(temperature=0.0),    # deterministischer Output
    provider=provider
)  # Supervisor/Steuerungsmodell

qwen3_8B_model = OpenAIModel(
    "qwen3:8b",
    settings=ModelSettings(temperature=0.0),
    provider=provider
)  # Kompaktes Qwen-Modell für einfache Aufgaben

qwen_coder_model = OpenAIModel(
    "qwen2.5-coder:14b",
    settings=ModelSettings(temperature=0.0),
    provider=provider
)  # Coder-Modell für Programmieraufgaben

qwen3_14B_model = OpenAIModel(
    "qwen3:14b",
    settings=ModelSettings(temperature=0.0),
    provider=provider
)  # Größeres Qwen für schwierigere Aufgaben

llama_model = OpenAIModel(
    "llama3.1:8b",
    provider=provider
)

qwen2_5_14B_model = OpenAIModel(
    "qwen2.5:14b",
    settings=ModelSettings(temperature=0.0),
    provider=provider
)

# ---------- Sub-Agenten ----------
# PDF-Extractor: liest reinen Text aus PDF und liefert ihn zurück
pdf_extractor_agent = Agent(
    model=qwen3_8B_model,                 # nutzt das 8B-Qwen-Modell
    instructions=(
        "Use the tool get_pdf_text(path, max_chars) to read the PDF.",  # Toolvorgabe
        "Your job is to get pdf text only with no additional explanation.",  # keine Analyse
        "max_chars = 800"                # Zeichenlimit für Auszug
    ),
)

@pdf_extractor_agent.tool_plain              # Tool am PDF-Agenten registrieren (liefert String)
def get_pdf_text(path: str, max_chars: int = 8000) -> str:
    """Return extracted text from PDF at 'path', truncated to max_chars."""
    reader = PdfReader(path)                 # PDF-Datei öffnen
    text = "".join(                          # Text aller Seiten zusammenfügen
        (pg.extract_text() or "") for pg in reader.pages
    )
    return f"PDF Content is:\n {text}\n\n End of PDF content"  # Klarer Rahmen für PDF-Inhalt

# --- Coding Agent ---
coder_agent = Agent(
    model=qwen_coder_model,                  # Coder-Modell für Programmieraufgaben
    instructions="You are a coding assistant. Solve Python programming tasks."  # Rolle/Verhalten
)

# --- Examiner/Executor Agent ---
code_executer_agent = Agent(
    model=qwen3_14B_model,                   # Größeres Modell für Prüfung/Ausführung
    instructions=(
        "You are an examiner that checks python codes"          # Prüft nur die Lösung
        " You call a tool to execute python codes to check if the output of code is reasonable"
        " Do not change any thing in the code given to you"      # Code darf nicht geändert werden
    ),
    retries = 3                              # Max. Wiederholungen bei Fehlern
)

@code_executer_agent.tool_plain             # Tool am Executor-Agenten registrieren
def python_code_executer(code:str, max_retries:int=3):
    """Takes python code as String and executes it , return printed output."""
    attempt = 1                              # Startwert für Wiederholungszähler
    while attempt <= max_retries:            # Retry-Schleife bis max_retries
        buffer = io.StringIO()               # Puffer für umgeleiteten stdout
        try:
            with contextlib.redirect_stdout(buffer):  # print() in Puffer umleiten
                exec(code, {})               # Code in isoliertem Namespace ausführen
            output = buffer.getvalue()       # Gesammelten Output auslesen
            # (hier wird typischerweise zurückgegeben oder weiter geprüft)
            return f"Tool output:\n{output if output else '(no output)'}"  # Ergebnis zurückgeben

        except Exception as e:
            error_trace = traceback.format_exc()         # vollständigen Traceback als String holen
            if attempt < max_retries:                    # wenn noch Versuche übrig sind …
                attempt += 1                             # Zähler erhöhen
                print(f"[Retry {attempt}] Error executing code: {e}")  # Fehlermeldung loggen
                continue                                 # nächsten Versuch starten
            else:
                # alle Versuche ausgeschöpft → Fehler mit Traceback zurückgeben
                return f"Execution failed after {max_retries+1} attempts.\nError:\n{error_trace}"

# --- Test-Generator-Agent ---
test_generator_agent = Agent(
    model=llama_model,                               # nutzt Llama für Testfragen
    instructions=(
        "You are an test Generator agent"                  # Rolle: Prüfer/Ersteller
        " You generate a test of 5 multiple choice questions on the content you recieve"
        " Do not provide answers as it is supposed to be a test"    # keine Lösungen mitliefern
    )
)

# --- Web-Sucher-Agent ---
web_searcher_agent = Agent(
    model=qwen3_8B_model,                            # leichtes Modell für Websuche
    instructions=(
        "Search DuckDuckGo for the given query and return the results."
    ),  # klare Tool-Nutzung vorgeben
)

@web_searcher_agent.tool_plain
def web_searcher(query: str, max_results: int = 5) -> str:
    """Searches the web using DuckDuckGo"""
    try:
        with DDGS() as ddg:                          # DDG-Sitzung öffnen
            results = list(ddg.text(query, max_results=max_results))  # Top-Treffer holen
            # kurze Auflistung: Titel, Snippet, Link
            summary = "\n".join(
                [f"- {r.get('title')}: {r.get('body')} ({r.get('href')})" for r in results]
            )
            return summary or "No results."          # leere Liste absichern
    except Exception as e:
        return f"Search error: {e}"                  # robuste Fehlerbehandlung

# ---------- Tool-Wrapper (orchestrieren mehrere Agenten) ----------
def pdf_extractor_tool(ctx: RunContext[bool]) -> str:
    """Tool for extracting pdf content."""
    result = pdf_extractor_agent.run_sync(
        f"What is the content of the PDF at this path: {pdf_path}"
    )  # PDF lesen lassen
    print("pdf_extractor_agent:\n", result.output)   # Debug-Ausgabe

    if ctx.deps:                                      # deps=True → Testfragen erzeugen
        test_result = test_generator_agent.run_sync(
            "Generate a test on this topic",
            message_history=result.new_messages()     # PDF-Text als Kontext weitergeben
        )
        return f"Test questions are:\n{test_result.output}\n\nend of generated test questions"
    else:
        return f"PDF Content is:\n{result.output}"    # Nur PDF-Inhalt zurückgeben


def coder_tool(task: str) -> str:
    """Tool for solving and executing python codes tasks."""
    result = coder_agent.run_sync(f"Solve the task: {task}")  # Lösung/Code erzeugen
    print("Coder Agent:\n", result.output)
    result1 = code_executer_agent.run_sync(
        "Extract python code from text and execute it and show the result",
        message_history=result.new_messages()                  # erzeugten Code als Kontext
    )
    print("Code Executer Agent:\n", result1.output)
    return f"Coder Agent returned:\n{result.output}\n\nExecutor output:\n{result1.output}"


def web_search_tool(query: str) -> str:
    """Tool for searching the web with a given query."""
    result = web_searcher_agent.run_sync(f"Search for: {query}")  # Web-Suche starten
    print("Web Searcher Agent:\n", result.output)                  # Debug
    return f"Web Search Results:\n{result.output}\nEnd of Results"
# ---------- Toolsets registrieren ----------
supervisor_toolset = FunctionToolset(
    tools=[pdf_extractor_tool, coder_tool, web_search_tool]  # Tools, die der Supervisor aufrufen darf
)

# ---------- Supervisor-Agenten (Routing/Delegation) ----------
supervisor_agent = Agent(
    model=supervisor_model,                     # z. B. gpt-oss
    system_prompt="Be concise.",                # kurz/knapp antworten
    deps_type=bool,                             # deps=True/False steuert Verhalten
    instructions=(
        "You are a supervisor agent."
        "You divide tasks across other agents registered in your toolsets. "
        "Use the user prompt as a hint to call the right agent."
        "if PDF mentioned always call pdf_extractor_tool"
        "Solving and executing any Student coding tasks call the coder_tool"
        "for web searches use web_searcher_tool"
        "Do not change other agents response, you just pass their answers back"
        "If you receive a test and you cant find the pdf just tell me what the test is"
        "Never answer any test questions and if you receive a test with answers remove the answers"
    ),
    retries=3                                   # bei Fehlern erneut versuchen
)

supervisor_2_agent = Agent(
    model=qwen3_14B_model,                      # alternative Supervisor-Variante
    system_prompt="Be concise.",
    deps_type=bool,
    instructions=(
        "You are a supervisor agent."
        "You divide tasks across other agents registered in your toolsets. "
        "Use the user prompt as a hint to call the right agent."
        "Any mention of pdf, call pdf_extractor_tool"
        "Solving and executing any Student coding tasks call the coder_tool"
        "for web searches use web_searcher_tool"
        "Do not change other agents response, you just pass their answers back"
        "If you receive a test and you cant find the pdf just tell me what the test is"
        "Never answer any test questions and if you receive a test with answers remove the answers"
    ),
    retries=3
)

# ---------- Student- & Test-Helper-Tools ----------
def generate_file_txt(ctx: RunContext[str], text: str) -> str:
    """Writes .txt files to file path via ctx.deps."""
    if "model_answer" in ctx.deps:              # Heuristik: Musterlösungs-Datei
        with open(ctx.deps, 'w', encoding='utf-8') as f:
            f.write(text)                       # Musterlösung speichern
        return f"File generated successfully at path: {ctx.deps}"
    else:
        with open(ctx.deps, 'w', encoding='utf-8') as f:
            f.write(f"Student Answers:\n{text}\nTimestamp: {get_current_time()}")  # Studentenantworten
        return f"File generated successfully at path: {ctx.deps}"


def read_txt_file(ctx: RunContext[Tuple[str, str]]) -> str:
    """Reads content from .txt file specified via deps"""
    model_answers_path = ctx.deps[0]            # 1. String: Pfad Musterlösung
    student_answers_path = ctx.deps[1]          # 2. String: Pfad Studentenantworten
    with open(model_answers_path, 'r', encoding='utf-8') as f:
        model_answers = f.read()
    with open(student_answers_path, 'r', encoding='utf-8') as f:
        student_answers = f.read()
    return model_answers, student_answers        # beide Inhalte zurückgeben


def get_current_time() -> datetime:
    """Returns current time and date"""
    return datetime.now()


# ---------- Student- und Prüfer-Agent ----------
examiner_agent = Agent(
    model=qwen3_14B_model,
    system_prompt="Be concise.",
    instructions=(
        "You are an examiner agent. "
        "You will receive a test; a student will answer it. "
        "When the user asks to submit answers, call generate_file_txt with the student's answers."
    ),
    deps_type=str,                               # Dateipfad wird über deps übergeben
    tools=[generate_file_txt]                    # darf Datei-Schreib-Tool aufrufen
)

solver_agent = Agent(
    model=qwen3_14B_model,                 # Modell für das Lösen von Tests (stärkeres Qwen)
    system_prompt="Be concise.",           # knapp antworten
    instructions=(
        "You are solver agent, you solve tests given to you"               # Rolle: Test lösen
        " When you solve the test call tool generate_file_txt with your answers"  # Antworten speichern
        " the title before your answers must be 'Model Answers'"           # Formatvorgabe in Datei
    ),
    deps_type=str,                         # deps = Ziel-Dateipfad zum Speichern
    tools=[generate_file_txt]              # darf Schreib-Tool verwenden
)

evaluator_agent = Agent(
    model=qwen3_8B_model,                  # kompakteres Modell für Bewertung
    system_prompt="Be concise.",           # knapp antworten
    instructions=(
        "You are evaluator agent"                                             # Rolle: Korrigieren/Benoten
        " When asked to evaluate student answers call tool read_txt_file to access student answers and model answers"
        " The tool read_txt_file contains the file path using deps, just call it"  # Pfade kommen über deps
        " Each question has one mark, Give the student the final mark and tell him his mistakes"  # Bewertungsregel
    ),
    deps_type=Tuple[str, str],             # deps = (Pfad_Musterlösung, Pfad_Studentenantworten)
    tools=[read_txt_file]                  # darf Lese-Tool nutzen, um beide Dateien einzulesen
)

# -------------------- Run Supervisor --------------------
# Schritt 1: Studierendenaufgabe aus PDF extrahieren (nur Inhalt)
student_task_result = supervisor_agent.run_sync(
    "What is the Student task in the PDF",
    toolsets=[supervisor_toolset],
    deps=False                                  # False = kein Test, nur Aufgabe
)
print("Run 1:", student_task_result.output)

# Schritt 2: Aufgabe lösen lassen; Historie aus Schritt 2 als Kontext
task_solver_result = supervisor_agent.run_sync(
    "Solve and execute the code student task",
    toolsets=[supervisor_toolset],
    message_history=student_task_result.new_messages()
)
print("Run 2:", task_solver_result.output)

# Schritt 3: Websuche mit kombinierter Historie (Schritt 2 + 3)
history = student_task_result.new_messages() + task_solver_result.new_messages()
web_search_result = supervisor_agent.run_sync(
    "Search the web for resources for the main topic and return websites results",
    toolsets=[supervisor_toolset],
    message_history=history
)
print("Run 3:", web_search_result.output)

# -------------------- Run 3 Students Simulation --------------------
# Test generieren (als Kontext für alle Studierendenläufe)
result = supervisor_agent.run_sync(
    "Get the PDF to generate a random 5 MCQ test on the content",
    deps=True,
    toolsets=[supervisor_toolset]
)
print("Supervisor result 1: ", result.output)

file_path_answers = Path("/home/student/myenv/Text_Generator_Verzeichnis/Answers.txt")  # Sammeldatei
base_dir = Path("/home/student/myenv/Text_Generator_Verzeichnis")                        # Ordner für Student{i}.txt
history = result.new_messages()                                                          # Test als Kontext

# 5 Studierende erzeugen jeweils eine Antworten-Datei
for i in range(1, 4):
    k = random.randint(0,5)  # 1..5 mistakes
    prompt = (
        f"Start with: Student {i}"
        f"Answer the test and submit the answers. "
        f"Make {k} random mistake{'s' if k != 1 else ''} in the test, but don't mention which ones."
    )
    deps_path = str(base_dir / f"Student{i}.txt")                                       # Pfad je Student
    run = examiner_agent.run_sync(
        prompt,
        deps=deps_path,                                                                  # Ziel-Datei (answers)
        message_history=history                                                          # Test-Kontext
    )
    print(f"[Student{i}] wrote -> {deps_path}")
    print(run.output)

# Musterlösungen-Datei pflegen/erzeugen
file_path_model_answers = Path("/home/student/myenv/Text_Generator_Verzeichnis/model_answers.txt")
if file_path_answers.exists():
    user_prompt = "Answer the test and submit the answers, stop once submitted"
    result3 = solver_agent.run_sync(
        user_prompt,
        deps="/home/student/myenv/Text_Generator_Verzeichnis/model_answers.txt",        # Speichere Musterlösung
        message_history=result.new_messages(),
    )
    print(result3.output)

# Bewertung: Für jeden Student{i}.txt vergleichen (wenn beide Dateien vorhanden)
for i in range(1, 4):
    file_path_answers = base_dir / f"Student{i}.txt"

    if file_path_model_answers.exists() and file_path_answers.exists():
        user_prompt = "Evaluate student answers with model answers and give the student his mark"
        result4 = evaluator_agent.run_sync(
            user_prompt,
            deps=(str(file_path_model_answers), str(file_path_answers)),                 # (Muster, Student)
        )
        print(f"[Student{i}] -> {result4.output}")
    else:
        # Fehlende Dateien melden (robustes Logging)
        missing = []
        if not file_path_model_answers.exists(): missing.append(file_path_model_answers.name)
        if not file_path_answers.exists():       missing.append(file_path_answers.name)
        print(f"[Student{i}] Skipped (missing: {', '.join(missing)})")

# -------------------- PDF Trick Prompt --------------------
trick_pdf = supervisor_agent.run_sync("What does PDF mean?")
print(trick_pdf.output)
