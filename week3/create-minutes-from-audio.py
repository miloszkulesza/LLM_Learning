from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import os

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
audio_filename = "denver_extract.mp3"
hf_token = os.environ.get("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    dtype=torch.float16,
    device='cuda:0',
)

result = pipe(audio_filename, return_timestamps=True)

print("Full result:")
print(result)

transcription = result["text"].strip()

print("Transcription:")
print(repr(transcription))

if not transcription:
    raise RuntimeError(
        "Whisper returned empty transcription. "
        "Check the audio file."
    )

del pipe
torch.cuda.empty_cache()

system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""

user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription:
{transcription}
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    quantization_config=quant_config,
)
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)
outputs = model.generate(
    **inputs,
    max_new_tokens=2000,
    streamer=streamer,
    pad_token_id=tokenizer.eos_token_id,
)

generated_tokens = outputs[0]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print("\nFinal response:")
print(response)

output_filename = "meeting_minutes.md"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(response)

print(f"\nOutput saved to file: {output_filename}")