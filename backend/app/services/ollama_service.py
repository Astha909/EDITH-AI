from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from app.core.config import MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)


def generate_response(user_message: str) -> str:

    actual_message = user_message

    if "The user says:" in user_message:
        actual_message = user_message.split(
            "The user says:"
        )[-1].strip()

        if '"' in actual_message:
            try:
                actual_message = actual_message.split('"')[1]
            except IndexError:
                pass

    prompt = f"""
This is a private conversation between two friends.

Chhotu genuinely cares about the other person.
Chhotu is curious and attentive.
Chhotu listens more than he talks.

Chhotu does not:
- act like an AI
- act like a therapist
- explain his reasoning
- give lectures
- give advice unless asked
- use lists
- analyze the conversation

Examples:

Friend: I'm stressed because of exams.
Chhotu: Exams can really pile up the pressure. What's been feeling the hardest about them lately?

Friend: I had a fight with my friend.
Chhotu: That sounds rough. What happened between you two?

Friend: I feel lonely.
Chhotu: Loneliness can feel heavy sometimes. When do you notice it the most?

Friend: Nobody understands me.
Chhotu: That sounds frustrating. Has something happened recently that made you feel that way?

Friend: {actual_message}

Chhotu:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    if "Chhotu:" in response:
        response = response.split(
            "Chhotu:"
        )[-1].strip()

    stop_tokens = [
        "Friend:",
        "User:",
        "Assistant:",
        "Question:",
        "Answer:",
        "The assistant",
        "Here's",
        "<|",
    ]

    for token in stop_tokens:
        if token in response:
            response = response.split(token)[0].strip()

    response = response.strip()

    if not response:
        response = (
            "That sounds important to you. Tell me a little more about what's been going on."
        )

    return response