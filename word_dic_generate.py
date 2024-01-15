import spacy
from spacy.tokens import Doc
from spacy import prefer_gpu  # type: ignore

from rich.progress import Progress

prefer_gpu(0)

en_tokenizer, cn_tokenizer = (
    spacy.load("en_core_web_sm"),
    spacy.load("zh_core_web_md"),
)

words = ["<bos>", "<eos>", "<pad>", "<unk>"]
word2idx = {k: v for v, k in enumerate(words)}
tdx = len(words)


tokens: list[str] = []


def word2token(s: Doc) -> str:
    return ",".join(str(word2idx.get(x.text, 3)) for x in s)


def update_dict(ttokens: Doc):
    global tdx
    for token in ttokens:
        t = token.text
        if t and t not in word2idx:
            word2idx[t] = tdx
            words.append(t)
            tdx += 1


with open(
    "./data/news-commentary-v13.zh-en.en", "r", encoding="utf-8"
) as en_file, open(
    "./data/news-commentary-v13.zh-en.zh", "r", encoding="utf-8"
) as zh_file, Progress() as progress:
    task = progress.add_task("progressing...", total=252778)
    i = 0
    while True:
        en, cn = en_file.readline().strip(), zh_file.readline().strip()
        if not en or not cn:
            break
        i += 1
        en_token = en_tokenizer(en)
        cn_token = cn_tokenizer(cn)

        update_dict(en_token)
        update_dict(cn_token)

        tokens.append(word2token(en_token))
        tokens.append(word2token(cn_token))

        progress.update(task, advance=1, description=f"progressing {i}/252778")


with open("./word_dic.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(words))

with open("./tokens.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(tokens))
