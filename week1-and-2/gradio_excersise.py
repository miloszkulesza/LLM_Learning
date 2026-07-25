import gradio as gr
from litellm import completion

def send_message(user_message, model):
    prompt = [
        { "role": "system", "content": system_message },
        { "role": "user", "content": user_message }
    ]
    response = completion(model=model,
                      messages=prompt,
                      api_base="http://localhost:11434",
                      stream=True)
    result = ""
    for chunk in response:
        result += chunk.choices[0].delta.content or ""
        yield result

def choose_model(user_message, model):
    if model == "GPT":
        result = send_message(user_message, "ollama_chat/gpt-oss")
    elif model == "Llama":
        result = send_message(user_message, "ollama_chat/llama3.2")
    else:
        raise ValueError("Unknown model")
    yield from result


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

system_message = "You are a helpful assistant that responds in markdown without code blocks"

message_input = gr.Textbox(label="Your message:", info="Enter a message for GPT-OSS", lines=7)
message_output = gr.Markdown(label="Response:")
model_selector = gr.Dropdown(["GPT", "Llama"], label="Select model", value="GPT")

view = gr.Interface(
    fn=choose_model,
    title="GPT-OSS",
    inputs=[message_input, model_selector],
    outputs=[message_output],
    examples=[
        ["Explain the Transformer architecture to a layperson", "GPT"],
        ["Explain the Transformer architecture to an aspiring programmer", "Llama"]
    ],
    flagging_mode="never",
)

view.launch(js=force_dark_mode)