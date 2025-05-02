import sys
sys.path.append('/home/trukhinmaksim/src')

from json import load
import requests
import base64
from time import time, sleep
import asyncio
import aiohttp
import re
from random import random
import json
import logging
import subprocess
import os

from src.utils.CacheAdapter import FlatAdapter, EXP_END_OF_DATA
from src.utils.DatabaseConnect import CacheConnector

#30-04-25_snan_readme(403).log

logging.basicConfig(
    filename="/home/trukhinmaksim/src/logs/01-05-25_scan_readme_clone.log",
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


PORTION_SIZE = 10
SKIP = 0
DB_LINK = "mongodb://10.22.90.255:27020/"

CLONE_LOCATION = "/home/trukhinmaksim/src/data/cache_30-04-25/repo"

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
    stdout_str = stdout.decode().strip()
    stderr_str = stderr.decode().strip()

    if process.returncode == 0:
        # cloned successfully
        return destination
    else:
        #print(f"Error cloning {repo_url} to {destination}:")
        print("\n\n\n\nE\n\n\n\n")
        if stderr_str:

            print(f"  Error message: {stderr_str}")

            # Check for common error patterns to provide more informative messages
            if "Authentication required" in stderr_str or "Could not read from remote repository" in stderr_str and "Permission denied (publickey)" in stderr_str:
                print("  Repository requires authentication (e.g., private repository) or SSH key issue. Skipping.")
            elif "Repository not found" in stderr_str:
                print("  Repository not found. Skipping.")
            elif "fatal: unable to access" in stderr_str and "The requested URL returned error: 403" in stderr_str:
                print("  Cloning blocked (e.g., due to organization restrictions). Skipping.")
            else:
                print("  An unknown error occurred. Skipping.")  # Generic message for other errors

            return ""

async def readFromClone(url):
    try:
        path = await clone(url)
        content = ""
        #print(path)
        #input("\n\n\nWaiting\n\n\n")

        #return {"content" : content}
        for root, dirs, files in os.walk(path):
            for file in files:
                if "readme" in file.lower():
                    print(file)
                    with open(os.path.join(path, file), encoding = "utf-8") as readmeFile:
                        content = readmeFile.read()
                    break

        return {"content" : content}

    except Exception as exp:
        if str(exp) == "empty":
            return {"content" : ""}
        print(f"Clone failed ({url}), error: {str(exp)}")
        logging.info(f"Clone failed ({url}), error: {str(exp)}")
        raise exp

def clearRepo():
    print("\n\n\n\n\nClearing\n\n\n\n\n")
    for root, dirs, files in os.walk(CLONE_LOCATION):
        for dir in dirs:
            subprocess.run(["rm", "-r", os.path.join(root, dir)])

async def fetchWithClientSession(urls = list(), tokenRotateCount = 0):
    tasks = list()
    #clearRepo()
    for url in urls:
        #print(f"Prepare to fetch {url}")
        tasks.append(readFromClone(url))

    results = await asyncio.gather(*tasks)

    return [result["content"] for result in results]



async def main():
    print("Welcome")
    logging.info("Welcome")

    readmeAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
    print("Adapter created on /home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
    logging.info("Adapter created on /home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
    
    collection = CacheConnector(DB_LINK).collection("raw_readme_30-04-25")

    start = time()
    urlsLst = []
    ids = []

    #with open("/home/trukhinmaksim/src/data/cache_30-04-25/clone_urls.json", "r", encoding = "utf-8") as file:
    #    urlsLst = json.load(fp = file)
    
    #with open("/home/trukhinmaksim/src/data/cache_30-04-25/ids.json", "r", encoding = "utf-8") as file:
    #    ids = json.load(fp = file)

    print(f"urlsLst = {urlsLst[:5]}")
    print(f"ids = {ids[:5]}")
    if len(ids) != len(urlsLst):
        print("Different len! Exiting")
        return

    i = 0
    counter = 0
    totalCounter = SKIP
    skip = SKIP
    start = time() #len(urlsLst)

    #await clearRepo()
    #exit()

    try:
        print(f"Starting from {skip}")
        logging.info(f"Starting from {skip}")

        urls = []
        ids = []

        while True:
            try:
                urls = []
                ids = []
                clearRepo()

                while len(urls) < PORTION_SIZE:
                    try:
                        item = readmeAdapter.load(1)[0]
                        if len(item["readme"]) == 0: # readme missing, prepare to read it
                            urls.append("https://api.github.com/repos/" + item["proj_id"][7:])
                            ids.append(item["proj_id"])


                    except EXP_END_OF_DATA:
                        if len(urls):
                            break # if something was already marked to fe
                        else:
                            raise EXP_END_OF_DATA

                print("Clonning repos")
                start = time()
                results = await fetchWithClientSession(urls = urls)
                d = zip(ids, results)
                for proj_id, readme in d:
                    collection.update_one({"proj_id" : proj_id}, {"proj_id" : proj_id, "readme" : readme})

                print(f"{PORTION_SIZE} repos were parsed in {time() - start}")

                if input(">>> ") != "c": break

            except EXP_END_OF_DATA:
                break

    except Exception as exp:
        print(f"Got exception during execution:", str(exp))
        logging.info(f"Got exception during execution:", str(exp))
        raise exp
    finally:
        #print(*results, sep = "\n" * 4)
        print(f"Saved {counter} readme files during execution, saved in total: {totalCounter}")
        print(f"Process completed in {time() - start}")
        logging.info(f"Saved {counter} readme files during execution, saved in total: {totalCounter}\nProcess completed in {time() - start}")



#text = fetchReadmeFile(url)
asyncio.run(main())
