import spacy
from spacy.tokens import Doc
from spacy import prefer_gpu  # type: ignore

prefer_gpu(0)

en_tokenizer, cn_tokenizer = (
    spacy.load("en_core_web_sm"),
    spacy.load("zh_core_web_md"),
)

words = ["<bos>", "<eos>", "<pad>", "<unk>"]
word2idx = {k: v for v, k in enumerate(words)}
tdx = len(words)
i = 0


def word2token(s: Doc) -> str:
    return ",".join(str(word2idx.get(x.text, 3)) for x in s)


tokens: list[str] = []

with open("./words copy.txt", "r", encoding="utf-8") as file:
    while True:
        en, cn = file.readline().strip(), file.readline().strip()
        if not en or not cn:
            break
        i += 1
        print(i)
        en_token = en_tokenizer(en)
        cn_token = cn_tokenizer(cn)

        for token in en_token:
            t = token.text.strip()
            if t and t not in word2idx:
                word2idx[t] = tdx
                words.append(t)
                tdx += 1

        for token in cn_token:
            t = token.text.strip()
            if t and t not in word2idx:
                word2idx[t] = tdx
                words.append(t)
                tdx += 1

        tokens.append(word2token(en_token))
        tokens.append(word2token(cn_token))


with open("./word_dic.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(words))

with open("./tokens.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(tokens))
