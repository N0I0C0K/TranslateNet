import spacy
from pprint import pprint

nlp = spacy.load("zh_core_web_md")

while (s := input(">")) != "exit":
    pprint(list(x.text for x in nlp(s)))
