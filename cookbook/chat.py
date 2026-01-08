#!/usr/bin/env python3
"""Simple terminal chat interface for gpt-oss-120b."""
import sys
from llm import LLM

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./gpt-oss-120b"
    print(f"Loading model from {model_path}...")

    try: model = LLM(model_path, max_num_seqs=10, max_seq_len=1024)
    except Exception as e: print(f"Error loading model: {e}", file=sys.stderr); sys.exit(1)
    print("Model loaded! Type 'quit' to exit.\n")

    messages = []
    try:
        while True:
            try: user_input = input("YOU: ").strip()
            except (EOFError, KeyboardInterrupt): print("\nGoodbye!"); break
            if user_input.lower() in ['quit', 'exit', 'q']: print("Goodbye!"); break
            if not user_input: continue

            messages.append({"role": "user", "content": user_input})
            try:
                result = model.chat(messages, max_tokens=1000).result()[0]
                print("ASSISTANT: " + result.text)
                if result.reasoning: print("REASONING: " + result.reasoning)
                messages.append({"role": "assistant", "content": result.text})
            except Exception as e: print("ERROR: " + str(e), file=sys.stderr); messages.pop()
    finally: model.stop()

if __name__ == "__main__": main()
