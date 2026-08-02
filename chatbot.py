import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/DialoGPT-medium"


def load_model():
    print(f"Loading tokenizer and model '{MODEL_NAME}' onto {DEVICE.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_response(
    user_input, chat_history_ids, tokenizer, model, max_context_tokens=512
):
    """Encodes user prompt, appends to history, and runs causal generation."""
    # Tokenize input text and append EOS token
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt"
    ).to(DEVICE)

    # Append to context buffer if history exists
    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    # Truncate context window to max tokens
    if bot_input_ids.shape[-1] > max_context_tokens:
        bot_input_ids = bot_input_ids[:, -max_context_tokens:]

    attention_mask = torch.ones(bot_input_ids.shape, device=DEVICE)

    # Generate response
    with torch.no_grad():
        updated_history_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.75,
        )

    # Extract target response sequence
    response_tokens = updated_history_ids[:, bot_input_ids.shape[-1] :]
    response_text = tokenizer.decode(
        response_tokens[0], skip_special_tokens=True
    )

    return response_text, updated_history_ids


def run_chatbot_demo():
    tokenizer, model = load_model()

    print("\n" + "=" * 60)
    print("      INTERACTIVE CONVERSATIONAL AI CHATBOT INTERFACE      ")
    print("=" * 60)

    sample_dialogue = [
        "Hello! Can you help me prepare for my computer science exam?",
        "What topics should I focus on for Deep Learning?",
        "Thank you so much for the advice!",
    ]

    chat_history = None
    print("\n--- RUNNING MULTI-TURN CONTEXT EVALUATION ---")

    for user_prompt in sample_dialogue:
        print(f"\nUser : {user_prompt}")
        bot_reply, chat_history = generate_response(
            user_prompt, chat_history, tokenizer, model
        )
        print(f"Bot  : {bot_reply}")

    print("\n" + "=" * 60)
    print("--- DEMO COMPLETED SUCCESSFULLY ---")
    print("=" * 60)


if __name__ == "__main__":
    run_chatbot_demo()