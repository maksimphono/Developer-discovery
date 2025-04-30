import sys
sys.path.append('/home/trukhinmaksim/src')

from json import load
import requests
import base64
from time import time, sleep
import asyncio
import aiohttp
import markdown
import re
from lxml import html
from random import random
import json

from src.utils.CacheAdapter import FlatAdapter

TOKEN = ""

with open("/home/trukhinmaksim/src/environment.json") as file:
    TOKENS = load(fp = file)["GITHUB_TOKENS"]

tokens = TOKENS
currentToken = tokens[0]
PORTION_SIZE = 10
SKIP = 16590 + 16030 + 1570 + 2830
INITIAL_TOKENS_ROTATION = 4

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

i = 0

async def fetch(session, url, retryingAttempt = 0):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                try:
                    content = (await response.json())["content"]
                    content = decodeFile(content)
                    return {"content" : content, "remain" : int(response.headers["X-RateLimit-Remaining"]), "exhausted" : False}
                except KeyError:
                    return {"content" : "", "remain" : int(response.headers["X-RateLimit-Remaining"]), "exhausted" : False}

                #print(f"{url} fetched, i = {i}")

            else:
                #print(f"Got response code = {response.status}")
                if response.status == 403 or response.status == 429: # rate limit exceeded
                    print(f"Got response code = {response.status}")
                    return {"content" : "", "remain" : 0, "exhausted" : True, "reset" : int(response.headers["X-RateLimit-Reset"])}
                if response.status == 404:
                    print(f"Not found: {str(url)}")
                return {"content" : "", "remain" : 0, "exhausted" : False, "remain" : int(response.headers["X-RateLimit-Remaining"])}

                #raise Exception("Error while fitch")
            #return await response.json()
    except TimeoutError as err:
        if retryingAttempt < 3:
            t = int(5 * (retryingAttempt + 1) + random() * 15)
            print(f"TimeoutError catched ({retryingAttempt}), retrying in {t}")
            await asyncio.sleep(t)
            return await fetch(session, url, retryingAttempt + 1)
        else:
            raise err

async def rotateTokens(enableSleep = True):
    tokens.append(tokens.pop(0))
    print(f"Rotate tokens; token : {tokens[0]}")

    if enableSleep:
        await asyncio.sleep(int(12 + random() * 15))

    return tokens



async def fetchWithClientSession(tokens, urls = list()):
    currentToken = tokens[0]
    tasks = list()
    async with aiohttp.ClientSession(headers={"Authorization": f"token {currentToken}"}) as session:
        #tasks = _tasks
        for url in urls:
            #print(f"Prepare to fetch {url}")
            tasks.append(fetch(session, url + "/contents/README.md"))

        results = await asyncio.gather(*tasks)

        if results[-1]["remain"] < PORTION_SIZE: # if the token will be exhausted on the next portion
            if results[-1]["exhausted"]:
                # token exhausted, wait until it resets and try again
                print(f"Token will reset in {results[-1]['reset'] - time()} s, sleeping...")
                asyncio.sleep(abs(results[-1]["reset"] - time()) + 10)
                return await fetchWithClientSession(tokens, urls) # try again after sleep (potentially token must reset during that time)
            else:
                await rotateTokens()

        return [result["content"] for result in results]


async def main():
    print("Welcome")

    readmeAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
    print("Adapter created on /home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
    start = time()
    urlsLst = []
    ids = []

    with open("/home/trukhinmaksim/src/data/cache_30-04-25/urls.json", "r", encoding = "utf-8") as file:
        urlsLst = json.load(fp = file)
    
    with open("/home/trukhinmaksim/src/data/cache_30-04-25/ids.json", "r", encoding = "utf-8") as file:
        ids = json.load(fp = file)

    print(f"urlsLst = {urlsLst[:5]}")
    print(f"ids = {ids[:5]}")
    if len(ids) != len(urlsLst):
        print("Different len! Exiting")
        return

    for j in range(INITIAL_TOKENS_ROTATION):
        await rotateTokens(False)

    print(tokens[0])
    i = 0
    counter = 0
    skip = SKIP
    start = time() #len(urlsLst)
    try:
        print(f"Starting from {skip}")
        for j in range(skip, len(urlsLst), PORTION_SIZE):
            results = await fetchWithClientSession(tokens, urls = urlsLst[j:j + PORTION_SIZE])
            d = zip(ids[j:j + PORTION_SIZE], results)
            #print([r[:10] for r in results])
            readmeAdapter.save([{"proj_id" : proj_id, "readme" : readme} for proj_id, readme in d])
            counter += len(results)

            if j % 1000 == 0:
                print(f"Scaned {j + PORTION_SIZE} repos; last: ({urlsLst[j + PORTION_SIZE - 1]}); token : {tokens[0]}")
                print(f"Content: {results[-1][:20]}\n")
                print(f"Total {counter} readme files were saved in {time() - start} s")

    except Exception as exp:
        print(f"Got exception during execution:", str(exp))
        raise exp
    finally:
        #print(*results, sep = "\n" * 4)
        print(f"Saved {counter} readme files")
        print(f"Process completed in {time() - start}")


url = f"https://api.github.com/repos/{owner}/{repo}"

#text = fetchReadmeFile(url)
asyncio.run(main())
