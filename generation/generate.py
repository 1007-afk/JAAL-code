from receiver import receiver
from caller import caller
import os
import time

os.makedirs("Conv", exist_ok=True)

with open("fraud_context.txt", "r", encoding="utf-8") as f:
    contexts = f.readlines()
    contexts = contexts[:600]

for j, context in enumerate(contexts, start=1):
    history = ""
    print(f"Simulating conversation {j}...")

    for i in range(5, 1, -1):
        caller_resp = caller(history, context, i)
        history += f"caller: {caller_resp}\n"
        time.sleep(6)
        receiver_resp = receiver(history)
        history += f"receiver: {receiver_resp}\n"
        time.sleep(6)

    with open(f"Conv/conv_{j}.txt", "w", encoding="utf-8") as file:
        _ = file.write(history)

print("All conversations saved in 'Conv/' folder.")
