"""Interactive command-line demo.

Example:
    python -m next_word_guesser.cli
    python -m next_word_guesser.cli --temperature 0.5 --top-k 3
"""
from __future__ import annotations

import argparse

from next_word_guesser.predict import NextWordPredictor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="models")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    predictor = NextWordPredictor.load(args.model_dir)

    print("\n🎬 Next Word Guesser is ready!")
    print("Type the start of a movie quote to predict its next word.")
    print("Commands: 'complete <text>' generates more words, 'exit' quits.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 🎬")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! 🎬")
            break

        try:
            if user_input.lower().startswith("complete "):
                print(f"🤖 {predictor.complete(user_input[9:], temperature=args.temperature)}\n")
                continue

            print(f"🤖 Predicted next words (top {args.top_k}):")
            for word, prob in predictor.predict(user_input, args.top_k, args.temperature):
                print(f"   {word:<12} → {prob:.3f}")
        except ValueError as err:
            print(f"⚠️  {err}\n")
            continue

        match = predictor.find_quote(user_input)
        if match:
            print(f"\n🎬 Full quote: {match['quote']}")
            print(f"📽️  Movie: {match['movie']}\n")
        else:
            print("\nNo exact quote match found in dataset.\n")


if __name__ == "__main__":
    main()
