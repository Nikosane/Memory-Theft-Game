import os
import random
import time
from model.learner import MemoryThief
from model.tokenizer import SimpleTokenizer
from utils.io_helpers import save_memory, load_memory

MEMORY_FILE = 'memory/memory_bank.json'
MODEL_FILE = 'memory/model_state.pth'


def intro():
    print("\n=== Welcome to Memory Theft Game ===\n")
    print("Talk naturally. The AI is learning your writing style...\n")


def chat_loop(ai_model, memory_bank, tokenizer):
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ('exit', 'quit'):
                print("\nExiting... Saving memory and model.\n")
                save_memory(MEMORY_FILE, memory_bank)
                ai_model.save_model(MODEL_FILE)
                break

            memory_bank.append(user_input)
            tokens = tokenizer.tokenize(user_input)
            ai_model.train_step(tokens)

            if random.random() < 0.2:
                print("(The AI is quietly learning...)\n")

            if len(memory_bank) > 10 and random.random() < 0.1:
                deception_test(ai_model, memory_bank, tokenizer)

    except KeyboardInterrupt:
        print("\n\nKeyboard Interrupt detected. Saving and exiting...")
        save_memory(MEMORY_FILE, memory_bank)
        ai_model.save_model(MODEL_FILE)


def deception_test(ai_model, memory_bank, tokenizer):
    print("\n=== Deception Test! ===")
    real_text = random.choice(memory_bank)
    fake_tokens = ai_model.generate_text()
    fake_text = tokenizer.detokenize(fake_tokens)

    choices = [real_text, fake_text]
    random.shuffle(choices)

    print("\nOne of these was written by YOU. One by the AI.")
    print("Can you tell which is which?\n")

    for idx, text in enumerate(choices, 1):
        print(f"{idx}. {text}")

    guess = input("\nWhich one is yours? (1/2): ").strip()
    if choices[int(guess) - 1] == real_text:
        print("\nCorrect! You still know yourself.\n")
    else:
        print("\nWrong! The AI fooled you...\n")

    print("(Returning to chat...)\n")
    time.sleep(2)


if __name__ == "__main__":
    os.makedirs('memory', exist_ok=True)

    memory_bank = load_memory(MEMORY_FILE)
    tokenizer = SimpleTokenizer()
    ai_model = MemoryThief()

    if os.path.exists(MODEL_FILE):
        ai_model.load_model(MODEL_FILE)

    intro()
    chat_loop(ai_model, memory_bank, tokenizer)

