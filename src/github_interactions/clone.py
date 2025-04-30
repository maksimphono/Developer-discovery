from json import load
import requests
import base64
from time import time
import asyncio
import aiohttp
import markdown
import re
from lxml import html

TOKEN = ""

with open("/home/trukhinmaksim/src/environment.json") as file:
    TOKEN = load(fp = file)["GITHUB_TOKEN"]

TOKENS = [TOKEN]

def decodeFile(fileData : str) -> str:
    return base64.b64decode(fileData).decode('utf-8')

owner = "iamkun"
repo = "dayjs"

def fetchReadmeFile(repo_url : str) -> str:
    url = f"{repo_url}/contents/README.md"
    response = requests.get(url, headers={"Authorization": f"token {TOKEN}"})

    if response.status_code == 200:
        content = response.json()["content"]
        return decodeFile(content)
    else:
        raise Exception("Error while fitch")

# github:guanzhi/GmSSL

def extractFromMD(markdown_content):
	text = re.sub(r'^#+\s*', '', markdown_content, flags=re.MULTILINE)

	# Remove bold and italic markers (* and **)
	text = re.sub(r'\*\*', '', text)
	text = re.sub(r'\*', '', text)

	# Remove inline code (`)
	text = re.sub(r'`', '', text)

	# Remove blockquotes (lines starting with '>')
	text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

	# Remove unordered list markers (- or *)
	text = re.sub(r'^-+\s*', '', text, flags=re.MULTILINE)

	# Remove ordered list markers (numbers followed by .)
	text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)

	# Remove links (both inline and reference style)
	text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)	# Inline links
	text = re.sub(r'\[([^\]]+)\]\[[^\]]+\]', r'\1', text)		# Reference links (without resolving)
	text = re.sub(r'^\[[^\]]+\]:\s.+', '', text, flags=re.MULTILINE) # Link definitions

	# Remove images (similar to links)
	text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'\1', text)
	text = re.sub(r'!\[([^\]]*)\]\[[^\]]+\]', r'\1', text)

	# Remove horizontal rules (---, ___, ***)
	text = re.sub(r'^-{3,}\s*$', '', text, flags=re.MULTILINE)
	text = re.sub(r'^_{3,}\s*$', '', text, flags=re.MULTILINE)
	text = re.sub(r'^\*{3,}\s*$', '', text, flags=re.MULTILINE)

	# Remove code blocks (```) - simple removal, might leave surrounding text
	text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

	# Remove HTML tags (if any are present in the Markdown) - basic removal
	text = re.sub(r'<[^>]+>', '', text)

	# Remove extra whitespace and newlines
	text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())

	return text

EXP_TOKEN_EXHAUSTED = Exception("TOKEN EXHAUSTED")

async def fetch(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            if response.headers["X-RateLimit-Remaining"] <= 0: 
                raise EXP_TOKEN_EXHAUSTED

            content = (await response.json())["content"]
            content = decodeFile(content)
            #content = extractFromMD(content)

            return content
        else:
            return ""
            #raise Exception("Error while fitch")
        #return await response.json()

currentToken = TOKENS[0]

async def fetchWithClientSession(tasks = list(), urls = list()):
    async with aiohttp.ClientSession(headers={"Authorization": f"token {currentToken}"}) as session:
        #tasks = _tasks
        for i, url in enumerate(urls[:1]):
            try:
                tasks.append(fetch(session, url + "/contents/README.md"))
            except Exception as exp:
                if exp is EXP_TOKEN_EXHAUSTED:
                    # switch token to the next one and continue
                    
                    currentToken = TOKEN
                    return await fetchWithClientSession(tasks = tasks, urls = urls[i:])

        #tasks = [fetch(session, url + "/contents/README.md") for url in urls[:1]]
        results = await asyncio.gather(*tasks)
        return results
        #print(*results, sep = "\n\n")
        #print(f"Time = {time() - start}")

async def main():
    print("main")

    start = time()
    urls = [
        "https://api.github.com/repos/spotify/scio",
        "https://api.github.com/repos/guanzhi/GmSSL",
        "https://api.github.com/repos/kn007/silk-v3-decoder",
        "https://api.github.com/repos/log4cplus/log4cplus",
        "https://api.github.com/repos/alibaba/MNN",
        "https://api.github.com/repos/react-native-image-picker/react-native-image-picker"
    ]

    start = time()
    results = await fetchWithClientSession(urls = urls)
    print(results)
    print(f"Time = {time() - start}")


url = f"https://api.github.com/repos/{owner}/{repo}"

#text = fetchReadmeFile(url)
asyncio.run(main())
