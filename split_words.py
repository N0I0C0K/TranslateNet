import spacy

from spacy import prefer_gpu  # type: ignore
from rich.progress import Progress

from concurrent.futures import ThreadPoolExecutor, Future

pool = ThreadPoolExecutor(32)


prefer_gpu(0)


def cn_tokenize():
    cn_tokenizer = spacy.load("zh_core_web_md")

    test = cn_tokenizer("初始化")

    def process_text(text: str) -> str:
        cn_token = cn_tokenizer(text)
        progress.update(task, advance=1, description=f"{i}/252778")
        return "/".join(map(lambda x: x.text, cn_token)) + "\n"

    with (
        open("./data/news-commentary-v13.zh-en.zh", "r", encoding="utf-8") as file,
        open("./data/news-commentary-v13.zh-en.zh.split", "w") as split_file,
        Progress() as progress,
    ):
        task = progress.add_task("progressing...", total=252778)
        i = 0
        while True:
            tasks: list[Future[str]] = []
            for _ in range(16):
                line = file.readline().strip()
                if len(line) == 0:
                    break
                tasks.append(pool.submit(process_text, line))

            if len(tasks) == 0:
                break
            i += len(tasks)
            progress.update(task, description=f"{i}/252778", advance=len(tasks))

            for fu in tasks:
                res = fu.result()
                split_file.write(res)


def en_tokenize():
    en_tokenizer = spacy.load("en_core_web_md")

    test = en_tokenizer("hello")

    def process_text(text: str) -> str:
        cn_token = en_tokenizer(text)
        progress.update(task, advance=1, description=f"{i}/252778")
        return "/".join(map(lambda x: x.text, cn_token)) + "\n"

    with (
        open("./data/news-commentary-v13.zh-en.en", "r", encoding="utf-8") as file,
        open("./data/news-commentary-v13.zh-en.en.split", "w") as split_file,
        Progress() as progress,
    ):
        task = progress.add_task("progressing...", total=252778)
        i = 0

        while True:
            tasks: list[Future[str]] = []
            for _ in range(16):
                line = file.readline().strip()
                if len(line) == 0:
                    break
                tasks.append(pool.submit(process_text, line))

            if len(tasks) == 0:
                break

            for fu in tasks:
                res = fu.result()
                split_file.write(res)
                i += 1
                progress.update(task, description=f"{i}/252778", advance=1)


en_tokenize()
