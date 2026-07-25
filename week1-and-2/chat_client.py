import gradio as gr
import ollama

system_message = "You are a helpful assistant"

force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') != 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
refresh();
"""

def extract_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)

def chat(message, history):
    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]
    for history_message in history:
        messages.append(
            {
                "role": history_message["role"],
                "content": extract_text(history_message["content"]),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )
    stream = ollama.chat(model="qwen3.5",
                           messages=messages,
                           stream=True)
    response = ""
    for chunk in stream:
        response += chunk.message.content or ''
        yield response

gr.ChatInterface(fn=chat).launch(js=force_dark_mode)