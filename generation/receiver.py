from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["Gem_api1"])


def receiver(history: str) -> str:
    context = f"""Act as a person who just picked up a call from an unknown number. 
Your language must be Hinglish (natural mix of Hindi and English).
History: {history}

Rules:
1. Output ONLY a valid JSON: "message": "your response".
2. Be organic: Use fillers like "uhm", "haan", "boliye".
3. No Placeholders: Do NOT use brackets like (name). If you don't know the caller's name, either ask "Aap kaun bol rahe ho?" or just assume a common name like "Rahul" or "Amit" to keep the flow.
4. If the history is empty, say something like "Hello? Kaun?"""
    resp = client.models.generate_content(
        model="gemma-3-27b-it", contents=context
    )
    return resp.text[24:-7]
