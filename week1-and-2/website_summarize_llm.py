import requests
from bs4 import BeautifulSoup
import ollama

MODEL = "llama3.2"

class Website:
    url: str
    text: str
    title: str

    def __init__(self, url: str):
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input", "nav", "footer", "header", "aside"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

system_prompt = "You are an assistant that analyzes the contents of a website and provides a short summary, ignoring text that might be navigation related. Respond in markdown."
def user_prompt_for(website):
    user_prompt = f"You are looking at a website titles {website.title}"
    user_prompt += "The contents of this website is as follows; please provide a short summary of this website in markdown. If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)},
    ]

def summarize(url):
    website = Website(url)
    messages = messages_for(website)
    response = ollama.chat(model=MODEL, messages=messages)
    print_statistics(response)
    return response['message']['content']

def display_summary(url):
    summary = summarize(url)
    print(summary)

def print_statistics(response):
    print(f"Prompt eval count: {response["prompt_eval_count"]}")
    print(f"Eval count: {response["eval_count"]}")
    print(f"Total duration: {response["total_duration"] / 1e9}")

display_summary("https://www.devtec.pl")