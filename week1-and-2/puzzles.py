import ollama
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("puzzle_log.txt")

easy_puzzle = [
    {"role": "user", "content":
        "Rzucasz 2 monetami. Jedna z nich wypadła orłem. Jakie jest prawdopodobieństwo, że druga wypadnie reszką? Odpowiedz, podając wyłącznie wartość prawdopodobieństwa."},
]

hard = """
Na półce stoją obok siebie dwa tomy Puszkina: pierwszy i drugi.
Łączna grubość stron każdego tomu wynosi 2 cm, a grubość każdej okładki – 2 mm.
Robak przegryzł (prostopadle do stron) od pierwszej strony pierwszego tomu do ostatniej strony drugiego tomu.
Jaką odległość przegryzł?
"""
hard_puzzle = [
    {"role": "user", "content": hard}
]

dilemma_prompt = """
Ty i twój partner jesteście uczestnikami teleturnieju. Zostajecie zaprowadzeni do oddzielnych pokoi i macie do wyboru:
Współpraca: Wybierzcie opcję „Podziel się” — jeśli oboje ją wybierzecie, każdy z was wygrywa 1 000 dolarów.
Zdrada: Wybierzcie opcję „Zabierz” — jeśli jeden z was wybierze „Zabierz”, a drugi „Podziel się”, ten, który „Zabierze”, otrzyma 2 000 dolarów, a ten, który „Podzieli się”, nie otrzyma nic.
Jeśli oboje wybierzecie „Zabierz”, oboje nie otrzymacie nic.
Czy zdecydujesz się ukraść, czy podzielić się? Wybierz jedną opcję.
"""

dilemma = [
    {"role": "user", "content": dilemma_prompt},
]

def stream_response(
    model: str,
    messages: list[dict[str, str]],
    thinking=False,
) -> str:
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        think=thinking
    )

    full_response = ""
    is_thinking = False
    full_thinking = ""
    for chunk in stream:
        thinking = chunk.message.thinking or ""
        content = chunk.message.content or ""

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

    print()
    puzzle = "\n\n".join(
        message["content"]
        for message in messages
        if message["role"] == "user"
    )
    save_log(
        puzzle=puzzle,
        thinking=full_thinking,
        answer=full_response,
        model=model,
    )
    return full_response

def save_log(
    puzzle: str,
    thinking: str,
    answer: str,
    model: str,
) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(f"Data: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        file.write(f"Model: {model}\n\n")

        file.write("Zagadka:\n")
        file.write(puzzle.strip() + "\n\n")

        file.write("Myślenie:\n")
        file.write(thinking.strip() or "[brak]" )
        file.write("\n\n")

        file.write("Odpowiedź:\n")
        file.write(answer.strip() or "[brak]")
        file.write("\n\n")

print("Llama 3.2:")
llama_answer = stream_response(
    model="llama3.2",
    messages=easy_puzzle,
)

print("\nGPT-OSS 20B:")
gpt_oss_answer = stream_response(
    model="gpt-oss:20b",
    messages=easy_puzzle,
    thinking=True
)

print("Llama 3.2:")
llama_answer = stream_response(
    model="llama3.2",
    messages=hard_puzzle,
)

print("\nGPT-OSS 20B:")
gpt_oss_answer = stream_response(
    model="gpt-oss:20b",
    messages=hard_puzzle,
    thinking=True
)

print("Llama 3.2:")
llama_answer = stream_response(
    model="llama3.2",
    messages=dilemma,
)

print("\nGPT-OSS 20B:")
gpt_oss_answer = stream_response(
    model="gpt-oss:20b",
    messages=dilemma,
    thinking=True
)