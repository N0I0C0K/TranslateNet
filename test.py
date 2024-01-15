import spacy
from pprint import pprint
import pyperclip

nlp = spacy.load("zh_core_web_md")

while (s := input(">")) != "exit":
    res = "/".join(x.text for x in nlp(s))
    pyperclip.copy(res)
    pprint(res)
