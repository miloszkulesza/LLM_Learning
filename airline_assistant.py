import json
import sqlite3
import gradio as gr
from openai import OpenAI

MODEL = "qwen3.5:4b"
openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
DB = "prices.db"

with sqlite3.connect(DB) as connection:
    cursor = connection.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)')
    connection.commit()

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

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

def get_ticket_price(destination_city):
    print(f"DATABASE TOOL CALL: Getting price for {destination_city}", flush=True)
    with sqlite3.connect(DB) as connection:
        cursor = connection.cursor()
        cursor.execute('SELECT price FROM prices WHERE city = ?', (destination_city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {destination_city} is ${result[0]}" if result else f"No price data available for this city"

def set_ticket_price(city, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO prices (city, price) VALUES (?, ?) ON CONFLICT(city) DO UPDATE SET price = ?',
                       (city.lower(), price, price))
        conn.commit()

ticket_prices = {"london":799, "paris": 899, "tokyo": 1420, "sydney": 2999}
for city, price in ticket_prices.items():
    set_ticket_price(city, price)

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]

def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses

def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses = handle_tool_calls(message)
        messages.append(message)
        messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    return response.choices[0].message.content

gr.ChatInterface(fn=chat).launch(js=force_dark_mode)