from google import genai

from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.environ["Gem_api1"])


def caller(history: str, call_context: str, responses: int) -> str:
    context = f"""Act as a caller with this specific context: {call_context}.
Current History: {history}
Turns remaining: {responses}

Rules:
1. Language: Hinglish. Tone should match the context (formal for bank, casual for friend).
2. Output ONLY a valid JSON: "message": "your response".
3. No Placeholders: Never use "(name)" or "[detail]". If a detail is missing, invent a realistic Indian name or detail on the spot. 
4. Organic Flow: If history is empty, start with a greeting like "Hello, kya meri baat Mrs. Sharma se ho rahi hai?"
5. Urgency: As {responses} gets lower, try to wrap up the conversation or reach the goal of the context. """
    resp = client.models.generate_content(
        model="gemma-3-27b-it", contents=context
    )
    return resp.text[24:-7]
