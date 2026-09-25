fraud_lines = []
with open("fraud_call.file", "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t", 1)  # split into label and text
        if len(parts) == 2 and parts[0].strip().lower() == "fraud":
            fraud_lines.append(parts[1].strip())

# print or save

for msg in fraud_lines:
    print(msg)

count = 0
with open("fraud_context.txt", "w") as f:
    for msg in fraud_lines:
        count+=1
        f.write(msg + "\n")
print(f"net : {count}")
