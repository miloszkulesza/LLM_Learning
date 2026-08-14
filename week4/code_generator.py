from openai import OpenAI
from system_info import retrieve_system_info
import subprocess
from pathlib import Path
import shutil
import os

MODEL = "qwen2.5-coder:3b"
openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

def setup_msvc():
    vcvars64 = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

    command = f'call "{vcvars64}" >nul && set'

    result = subprocess.run(
        command,
        shell=True,
        check=True,
        text=True,
        capture_output=True
    )

    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value

    cl = shutil.which("cl.exe")

    if not cl:
        raise RuntimeError("vcvars64.bat executed, but cl.exe was not found")

    print(f"MSVC compiler: {cl}")

setup_msvc()
system_info = retrieve_system_info()

message = f"""
Here is a report of the system information for my computer.
I want to run a C++ compiler to compile a single C++ file called main.cpp and then execute it in the simplest way possible.
Please reply with whether I need to install any C++ compiler to do this. If so, please provide the simplest step by step instructions to do so.

If I'm already set up to compile C++ code, then I'd like to run something like this in Python to compile and execute the code:
```python
compile_command = # something here - to achieve the fastest possible runtime performance
compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
run_command = # something here
run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
return run_result.stdout
```
Please tell me exactly what I should use for the compile_command and run_command.

System information:
{system_info}
"""

def send_prompt(messages):
    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)
    full_response = ""
    is_thinking = False
    full_thinking = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        thinking = getattr(delta, "thinking", None) or getattr(delta, "reasoning_content", None) or ""
        content = delta.content or ""

        if thinking:
            if not is_thinking:
                print("Thinking:\n", end="", flush=True)
                is_thinking = True
            print(thinking, end="", flush=True)
            full_thinking += thinking
        elif content:
            if is_thinking:
                print("\n\nAnswer:\n", end="", flush=True)
                is_thinking = False
            print(content, end="", flush=True)
            full_response += content
    return full_response

messages = [{"role": "user", "content": message}]
print(f"Calling LLM: {message}")
send_prompt(messages)

compile_command = [
    "cl.exe",
    "/nologo",
    "/EHsc",
    "/O2",
    "/GL",
    "/DNDEBUG",
    "/std:c++20",
    "main.cpp",
    "/Fe:main.exe",
    "/link",
    "/LTCG"
]
run_command = [str(Path.cwd() / "main.exe")]

system_prompt = """
Your task is to convert Python code into high performance C++ code.

STRICT OUTPUT RULES:
- Output ONLY valid C++ source code.
- Do NOT use Markdown.
- Do NOT use ```cpp or ``` fences.
- Do NOT provide explanations.
- Do NOT provide compilation commands.
- Do NOT write any text before or after the C++ source code.
- The first non-whitespace characters of your response must be valid C++ code.
- The output must compile as-is when saved directly to main.cpp.

The C++ program must produce output equivalent to the Python program.
Optimize primarily for runtime performance.
"""

def user_prompt_for(python):
    return f"""
Port this Python code to C++ with the fastest possible implementation that produces identical output in the least time.
The system information is:
{system_info}
Your response will be written to a file called main.cpp and then compiled and executed; the compilation command is:
{compile_command}
Respond only with C++ code.
Python code to port:

```python
{python}
```
"""

def messages_for(python):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(python)}
    ]

def write_output(cpp):
    with open("main.cpp", "w", encoding="utf-8") as f:
        f.write(cpp)

def port(python):
    print(f"Calling LLM: {messages_for(python)}")
    reply = send_prompt(messages_for(python))
    reply = reply.replace('```cpp','').replace('```','')
    write_output(reply)

pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""

def run_python(code):
    globals = {"__builtins__": __builtins__}
    exec(code, globals)

def compile_and_run():
    subprocess.run(compile_command, check=True, text=True, capture_output=True, shell=True)
    print(subprocess.run(run_command, check=True, text=True, capture_output=True, shell=True).stdout)
    print(subprocess.run(run_command, check=True, text=True, capture_output=True, shell=True).stdout)
    print(subprocess.run(run_command, check=True, text=True, capture_output=True, shell=True).stdout)

print("Running")
run_python(pi)
print("Porting...")
port(openai, MODEL, pi)
print("Compiling...")
compile_and_run()