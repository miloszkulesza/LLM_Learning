import gradio as gr
from litellm import completion

def message_gpt(user_message):
    prompt = [
        { "role": "system", "content": system_message },
        { "role": "user", "content": user_message }
    ]
    response = completion(model="ollama_chat/gpt-oss",
                      messages=prompt,
                      api_base="http://localhost:11434",
                      stream=True)
    result = ""
    for chunk in response:
        result += chunk.choices[0].delta.content or ""
        yield result

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

view = gr.Interface(
    fn=message_gpt,
    title="GPT-OSS",
    inputs=[message_input],
    outputs=[message_output],
    examples=[
        "Explain the Transformer architecture to a layperson",
        "Explain the Transformer architecture to an aspiring programmer"
    ],
    flagging_mode="never",
)

view.launch(js=force_dark_mode)