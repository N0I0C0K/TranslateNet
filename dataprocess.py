res = []
with open("./hsk_1_4.txt", "r") as file:
    i = 0
    while True:
        line = file.readline()
        if not line:
            break
        if i % 5 == 0:
            res.append(line[9:].strip())
        elif i % 5 == 2:
            res.append(line[10:].strip())
        i = (i + 1) % 5

with open("./words.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(res))
