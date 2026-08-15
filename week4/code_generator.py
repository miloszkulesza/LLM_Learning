from openai import OpenAI
from system_info import retrieve_system_info
import subprocess
from pathlib import Path
import shutil
import os
import gradio as gr
from styles import CSS

models = ["qwen2.5-coder", "deepseek-coder-v2", "gpt-oss:20b"]
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

def send_prompt(model, messages):
    stream = openai.chat.completions.create(model=model, messages=messages, stream=True)
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
You are an expert C++ performance engineer.
Convert the provided Python program into high-performance C++20.

Requirements:
- Preserve the observable output and behavior of the Python program.
- Optimize for execution speed.
- Target the provided hardware and MSVC compiler.
- Return only valid C++ source code.
- Do not use Markdown code fences.
- Do not include explanations or comments.
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

def port(model, python):
    print(f"Calling LLM: {messages_for(python)}")
    reply = send_prompt(model, messages_for(python))
    reply = reply.replace('```cpp','').replace('```','')
    write_output(reply)
    return reply

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

def compile_and_run():
    try:
        subprocess.run(compile_command, check=True, text=True, capture_output=True)

        output = subprocess.run(run_command, check=True, text=True, capture_output=True).stdout
        full_output = ''
        full_output += output
        print(output)

        output = subprocess.run(run_command, check=True, text=True, capture_output=True).stdout
        full_output += '\n' + output
        print(output)

        output = subprocess.run(run_command, check=True, text=True, capture_output=True).stdout
        full_output += '\n' + output
        print(output)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred:\n{e.stderr}")
        return f"Compilation/execution failed:\n{e.stderr}"

    return full_output

with gr.Blocks(css=CSS, theme=gr.themes.Monochrome(), title=f"Port from Python to C++") as ui:
    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            python = gr.Code(label="Python code:", language="python", lines=26, value=pi)
        with gr.Column(scale=6):
            cpp = gr.Code(label="C++ code:", language="cpp", lines=28)
    with gr.Row(elem_classes=["controls"]):
        model = gr.Dropdown(models, label="Select model", value=models[0])
        convert = gr.Button("Convert code", elem_classes=["convert-btn"])
        compile = gr.Button("Compile and run C++", elem_classes=["run-btn", "cpp"])
    with gr.Row(equal_height=True):
        output = gr.TextArea(label="Result:", lines=8, elem_classes=["cpp-out"])

    convert.click(port, inputs=[model, python], outputs=[cpp])
    compile.click(compile_and_run, outputs=[output])

ui.launch(inbrowser=True)