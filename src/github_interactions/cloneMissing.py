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
import logging
import subprocess
import os

from src.utils.CacheAdapter import FlatAdapter, EXP_END_OF_DATA
from src.utils.DatabaseConnect import CacheConnector

#30-04-25_snan_readme(403).log

logging.basicConfig(
    filename="/home/trukhinmaksim/src/logs/02-05-25_scan_missing_readme.log",
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

TOKEN = ""

with open("/home/trukhinmaksim/src/environment.json") as file:
    TOKENS = load(fp = file)["GITHUB_TOKENS"]

tokens = TOKENS
currentToken = tokens[0]
PORTION_SIZE = 20
SKIP = 0
INITIAL_TOKENS_ROTATION = 0
reposWithoutReadme = []

CLONE_LOCATION = "/home/trukhinmaksim/src/data/cache_30-04-25/repo"

def decodeFile(fileData : str) -> str:
    try:
        return base64.b64decode(fileData).decode('utf-8')
    except UnicodeDecodeError:
        return ""

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

EXP_TOKEN_EXHAUSTED = Exception("TOKEN EXHAUSTED")

i = 0

async def clone(repo_url):
    # executes teminal command to clone repository
    destination = os.path.join(CLONE_LOCATION, repo_url[19:].replace("/", "--"))
    print(f"Cloning {repo_url} to {destination}...")
    await asyncio.create_subprocess_exec("mkdir", destination)
    process = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth", "1", repo_url, destination,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        return destination
    else:
        #print(f"Error cloning {repo_url} to {destination}:")
        if stderr:
            raise Exception(stderr.decode())


async def readFromClone(url):
    try:
        path = await clone(url)
        content = ""

        for root, dirs, files in os.walk(path):
            for file in files:
                if "readme" in file.lower():
                    with open(os.path.join(path, file), encoding = "utf-8") as readmeFile:
                        content = readmeFile.read()
                    break

        return {"content" : content}

    except Exception as exp:
        print(f"Clone failed ({url}), error: {str(exp)}")

def handleBadResponse(response, url, message = ""):
    try:
        if response.status == 403 or response.status == 429: # rate limit exceeded
            logging.info(f"Got response code = {response.status} ({url})")
            print(f"Got response code = {response.status} ({url})")
            print({"reset" : int(response.headers["X-RateLimit-Reset"]), "message" : message})
            if message == "Repository access blocked":
                return {"content" : "", "remain" : int(response.headers["X-RateLimit-Remaining"]), "exhausted" : False, "reset" : int(response.headers["X-RateLimit-Reset"])}
            else:
                return {"content" : "", "remain" : 0, "exhausted" : True, "reset" : int(response.headers["X-RateLimit-Reset"])}
        if response.status == 404:
            print(f"Not found: {str(url)}")
            logging.info(f"Not found: {str(url)}")
        return {"content" : "", "remain" : 0, "exhausted" : False, "remain" : int(response.headers["X-RateLimit-Remaining"])}

    except KeyError:
        pass
        #return {"content" : "", "remain" : 0, "exhausted" : False, "remain" : int(response.headers["X-RateLimit-Remaining"])}


async def fetchReadmeUrl(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            for item in await response.json():
                if "readme" in item["name"].lower():
                    return (url + "/" + item["name"], None)

            return ("", {"content" : "", "remain" : int(response.headers["X-RateLimit-Remaining"]), "exhausted" : False, "reset" : int(response.headers["X-RateLimit-Reset"])})
        else:
            return ("", handleBadResponse(response, url, (await response.json())["message"]))



async def fetch(session, url, retryingAttempt = 0):
    try:
        readmeUrl, error = await fetchReadmeUrl(session, url)
        if error:
            return error

        async with session.get(readmeUrl) as response:
            if response.status == 200:
                try:
                    content = (await response.json())["content"]
                    content = decodeFile(content)
                    return {"content" : content, "remain" : int(response.headers["X-RateLimit-Remaining"]), "exhausted" : False, "reset" : int(response.headers["X-RateLimit-Reset"])}
                except KeyError:
                    return {"content" : "", "remain" : int(response.headers["X-RateLimit-Remaining"]), "exhausted" : False, "reset" : int(response.headers["X-RateLimit-Reset"])}
                except TypeError:
                    return {"content" : "", "remain" : int(response.headers["X-RateLimit-Remaining"]), "exhausted" : False, "reset" : int(response.headers["X-RateLimit-Reset"])}

                #print(f"{url} fetched, i = {i}")

            else:
                #print(f"Got response code = {response.status}")
                return handleBadResponse(response, readmeUrl, (await response.json())["message"])

                #raise Exception("Error while fitch")
            #return await response.json()
    except TimeoutError as err:
        if retryingAttempt < 3:
            t = int(5 * (retryingAttempt + 1) + random() * 15)
            print(f"TimeoutError catched ({retryingAttempt}), retrying in {t}")
            logging.info(f"TimeoutError catched ({retryingAttempt}), retrying in {t}")
            await asyncio.sleep(t)
            return await fetch(session, url, retryingAttempt + 1)
        else:
            raise err

async def rotateTokens(enableSleep = True):
    tokens.append(tokens.pop(0))
    print(f"Rotate tokens; token : {tokens[0]}")
    logging.info(f"Rotate tokens; token : {tokens[0]}")

    if enableSleep:
        await asyncio.sleep(int(12 + random() * 15))

    return tokens



async def fetchWithClientSession(tokens, urls = list(), tokenRotateCount = 0):
    currentToken = tokens[0]
    tasks = list()
    async with aiohttp.ClientSession(headers={"Authorization": f"token {currentToken}"}) as session:
        #tasks = _tasks
        for url in urls:
            #print(f"Prepare to fetch {url}")
            tasks.append(fetch(session, url + "/contents"))

        results = await asyncio.gather(*tasks)

        if any((result["remain"] < PORTION_SIZE * 2 for result in results)): # if the token will be exhausted on the next portion
            if any((result["exhausted"] for result in results)):
                # token exhausted, wait until it resets and try again
                if tokenRotateCount >= len(tokens):
                    print(f"Token will reset in {results[-1]['reset'] - time()} s, sleeping... Token : {tokens[0]}")
                    logging.info(f"Token will reset in {results[-1]['reset'] - time()} s, sleeping... Token : {tokens[0]}")
                    await asyncio.sleep(abs(results[-1]["reset"] - time()) + 10)
                    return await fetchWithClientSession(tokens, urls, 0) # try again after sleep (potentially token must reset during that time)
                else:
                    await rotateTokens()
                    return await fetchWithClientSession(tokens, urls, tokenRotateCount + 1) # try again after sleep (potentially token must reset during that time)
            else:
                await rotateTokens()

        return [result["content"] for result in results]


async def main():
    print("Welcome")
    logging.info("Welcome")

    readmeAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
    collection = CacheConnector("mongodb://10.22.48.31:27020/").collection("raw_readme_30-04-25")

    print("Adapter created on /home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
    logging.info("Adapter created on /home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
    start = time()
    urlsLst = []
    ids = []
    if len(ids) != len(urlsLst):
        print("Different len! Exiting")
        return

    for j in range(INITIAL_TOKENS_ROTATION):
        await rotateTokens(False)

    print(tokens[0])
    i = 0
    counter = 0
    totalCounter = 0
    skip = 0
    start = time() #len(urlsLst)
    cursor = collection.find({"readme" : ""})
    start = time()

    try:
        while True:
            try:
                urls = []
                ids = []

                while len(urls) < PORTION_SIZE:
                    try:
                        item = next(cursor)
                        urls.append("https://api.github.com/repos/" + item["proj_id"][7:])
                        ids.append(item["proj_id"])

                    except StopIteration:
                        if len(urls):
                            break # if something was already marked to fe
                        else:
                            raise EXP_END_OF_DATA

                results = await fetchWithClientSession(tokens, urls = urls)
                d = zip(ids, results)

                for proj_id, readme in d:
                    #print(proj_id, len(readme))
                    if readme == "":
                        reposWithoutReadme.append(proj_id)
                    else:
                        collection.update_one({"proj_id" : proj_id}, {"$set" : {"readme" : readme}})

                counter += len(results)

                if counter % 1000 == 0:
                    print(f"Collected missing readme: {counter}")
                    logging.info(f"Collected missing readme: {counter}")

                #print(f"{PORTION_SIZE} repos were parsed in {time() - start}")

                #if input(">>> ") != "c": break

            except EXP_END_OF_DATA:
                break

    except Exception as exp:
        print(f"Got exception during execution:", str(exp))
        logging.info(f"Got exception during execution:", str(exp))
        raise exp
    finally:
        #print(*results, sep = "\n" * 4)
        print(f"Saved {counter} readme files during execution, saved in total: {totalCounter}")
        logging.info(f"Saved {counter} readme files during execution, saved in total: {totalCounter}\nProcess completed in {time() - start}")

        with open("/home/trukhinmaksim/src/data/cache_30-04-25/repos_without_readme.json", "w", encoding = "utf-8") as file:
            json.dump(reposWithoutReadme, fp = file, ensure_ascii = False)

        print(f"Process completed in {time() - start}")
        

with open("/home/trukhinmaksim/src/data/cache_30-04-25/repos_without_readme.json", encoding = "utf-8") as file:
    reposWithoutReadme = json.load(fp = file)

#text = fetchReadmeFile(url)
asyncio.run(main())
