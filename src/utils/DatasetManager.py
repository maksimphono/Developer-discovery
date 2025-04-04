import sys
sys.path.append('/home/trukhinmaksim/src')

from numpy import array, ndarray
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from copy import deepcopy
import re
from collections import defaultdict
import argostranslate.package
import argostranslate.translate
from random import random
from time import sleep
from langdetect import detect
import requests
from json import dumps
from random import choice
import textstat

from src.utils.CacheAdapter import JSONAdapter
from src.data_processing.collect_projects_data import collectOneProjectData, EXP_NOT_IN_DB

def downloadArgosLangPackages(langList = ["es", "pt", "zh", "zt", "ru", "de", "ja", "ko"]):
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    packages = list(filter(lambda pkg: pkg.from_code in langList and pkg.to_code == "en", available_packages))
    print(packages)
    for pkg in packages:
        argostranslate.package.install_from_path(pkg.download())


class IgnoreList(dict):
    def includes(self, item):
        return self[item]

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            return False

    def extend(self, collection):
        if type(collection) in [dict, defaultdict]:
            self.update(collection)

        elif type(collection) in [list, ndarray, array]:
            for item in collection:
                self[item] = True

        else:
            raise Exception("method IgnoreList.extend must take dict, array or list")

    def add(self, item):
        self[item] = True

    def remove(self, item):
        del self[item]

    def __iter__(self):
        for item in self.keys():
            yield item


class ProjectsDatasetManager:
    usersCollection = None
    projectsCollection = None
    translatorServers = []

    def __init__(self, userNumber = float("inf"), validate = lambda data: True, cacheAdapter = None):
        self.userNumber = userNumber
        self.validate = validate
        self.data = None
        self.preprocessed = False
        self.ignoredUsers = IgnoreList()
        self.readability = {"flesch" : 13, "dale_chall" : 11}
        self.readabilityCheckEnabled = True
        if ProjectsDatasetManager.usersCollection != None:
            self.cursor = ProjectsDatasetManager.usersCollection.find()
        else:
            # not specified, then the cursor will be set later manually
            self.cursor = None 
        #download_CN_EN_ArgosPackage()
        
        if cacheAdapter == None: 
            self.cacheAdapter = JSONAdapter()
        else:
            self.cacheAdapter = cacheAdapter

    def resetCursor(self):
        self.cursor = ProjectsDatasetManager.usersCollection.find()
    
    def clearData(self):
        self.data.clear()
        self.preprocessed = False

    def ignoreUsers(self, users_ids : list[str] | str):
        if type(users_ids) == str:
            # path to the file with ignored users is specified
            with open(users_ids, encoding="utf-8") as file:
                self.ignoredUsers.extend(json.load(file))
        else:
            self.ignoredUsers.extend(users_ids)
    
    def fromCache(self):
        # loads excatly 'self.userNumber' users from cache per call
        self.data = self.cacheAdapter.load(self.userNumber)

        # it is assumed, that cache only contains already preprocessed data, so no need to perprocess or filter data, all done!
        self.preprocessed = True
        return self.data

    def fromDB(self):
        # loads excatly 'self.userNumber' users from database per call
        self.data = self.getProjectsDataForUsers()
        self.preprocessed = False # assume, that database contains unprocessed data
        return self.data

    def getProjectsDataForUsers(self) -> dict[str, list]:
        # will return a dictionary, where keys are users ids and values are lists of projects ids, each user has contributed to
        i = 0
        count = self.userNumber
        #cursor = ProjectsDatasetManager.usersCollection.find().skip(len(self.ignoredUsers))
        data = {}

        while count > 0:
            user = next(self.cursor) # cursor continues from the point, where it was left during the last time the method was called
            if self.ignoredUsers.includes(user["id"]):
                #print(f"{user['id']} ignored")
                continue # if that user must be ignored, just skip to the next one
            else:
                #print(f"Trying to scan {user['id']}")
                
                projectsIDList = user["projects"]

                projects = []

                for proj_id in projectsIDList:
                    #print(f"Searching project data {proj_id}")
                    try:
                        projectData = collectOneProjectData(proj_id)
                    except Exception as exp:
                        if exp is EXP_NOT_IN_DB: 
                            #print(f"{proj_id} not in db")
                            continue # if the project data wasn't found , just skip
                        else:
                            raise exp

                    #projectData = ProjectsDatasetManager.projectsCollection.find_one({"proj_id" : proj_id}, {"_id" : False})
                    #print("Found project data")
                    
                    if self.validate(projectData):
                        projectData["description"] = self.translateText(projectData["description"])

                        if self.readabilityCheckEnabled:
                            if self.checkTextReadability(projectData["description"]):
                                projects.append(projectData)
                        else:
                            projects.append(projectData)
        
                if len(projects):
                    # if user has at least one project he contributed to
                    print(f"Scanning {user['id']}")
                    data[user["id"]] = deepcopy(projects)
                    count -= 1

                i += 1

        return data

    def translateText(self, text, retry = 3, useServer = False):
        # will try to use Google Translate, but if any error occures, will use Argos offline translator
        if text.isascii() or detect(text) == "en": return text # if the text is already english (either ascii or english with unicode emoji)

        if useServer:
            translatorURL = choice(ProjectsDatasetManager.translatorServers)

            #print(translatorURL)
            response = requests.request("POST", url = translatorURL, headers = {'Content-Type': 'application/json'}, data = dumps({"text" : text}, ensure_ascii=False, indent=4))

            if response.ok:
                return response.json()["translate"]
        else:
            import asyncio
            import nest_asyncio
            from googletrans import Translator

            for i in range(retry):
                try:
                    async def inner():
                        nonlocal text

                        async with Translator() as translator:
                            result = await translator.translate(text, dest = "en")

                        return result

                    nest_asyncio.apply()  # Patch the event loop    
                    return asyncio.run(inner()).text

                except Exception as exp:
                    # assume, that the text is in Chinese and translate it using argos translator
                    sleep(random() * 6)
                    print("Conection error, retrying")
                    continue

        langCode = detect(text)[:2]
        if langCode not in ["es", "pt", "zh", "zt", "ru", "de", "ja", "ko"]: langCode = "zh" # If language is unknown, assuming it is Chinese
        print(f"Translation Faled after {retry} attempts, Using Argos for {text[:10]}...\nLanguage detected as: {langCode}")
        return argostranslate.translate.translate(text, langCode, "en")
        """
        if str(type(exp)) == "<class 'httpx.ConnectError'>":
            return text
        else:
            raise exp
        """

    def checkTextReadability(self, text):
        return (textstat.flesch_reading_ease(text) >= self.readability["flesch"] and textstat.dale_chall_readability_score(text) >= self.readability["dale_chall"])

    def textPreprocessing(self, text):
        # Initialize tools
        stop_words = set(stopwords.words("english") + ["etc"])
        lemmatizer = WordNetLemmatizer()

        # Translate:
        #text = self.translateText(text, 3)
        # Remove unicode:
        text = text.encode("ascii", "ignore").decode()
        # Process camel case:
        #text = processCamelCase(text)
        # Lower the text:
        text = text.lower()
        # remove links and urls:
        text = re.sub(r"http[^\s]*", "", text)
        # Remove punctuation:
        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        # Remove stop-words:
        #text = re.sub("\s" + "|".join(stop_words) + "\s", " ", text)
        # Remove numbers:
        text = re.sub(r"\d", " ", text)
        # Remove new lines:
        text = re.sub(r"\n", " ", text)
        # Remove multiple spaces:
        text = re.sub("\s+", " ", text).strip()
    
        tokens = [word for word in word_tokenize(text) if word not in stop_words and len(word) > 1]  # Tokenize into words
        
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Remove stopwords & lemmatize

        return tokens

    def projectsDataPreprocessing(self, projects : array(dict), includingText = False) -> array([{"tokens" : str, "tags" : list}]):
        # will take in an array of projects and prepare it to be consumed by the model
        # takes: array of projects (as dictionaries); returns: text data and tags for every project in array
        result = []

        for proj in projects:
            joinedText = " ".join([proj["name"], proj["description"]])

            tockens = self.textPreprocessing(joinedText)
            # only meaningfull tags will be saved, no empty strings!
            tags = list(filter(lambda n: (n != ""), [proj["id"], proj["name"], proj["language"]] + proj["topics"]))# if proj["language"] else proj["topics"]

            if includingText:
                result.append({"text" : joinedText, "tokens" : tockens, "tags" : tags})
            else:
                result.append({"tokens" : tockens, "tags" : tags})
                

        return result

    def preprocess(self, _data : dict | None = None, including_text : bool = False) -> dict[str, list]:
        if self.preprocessed: return self.data

        if _data:
            data = _data
        elif self.data:
            data = self.data
        else:
            return self.fromCache()

        for user_id, projs in data.items():
            #print(type(array(userProjs)))
            data[user_id] = self.projectsDataPreprocessing(projs, including_text)

        self.preprocessed = True
        return data

    def getTextOnly(self, _data : dict | None = None):
        result = {}

        if _data:
            data = _data
        elif self.data:
            data = self.data
        else:
            return self.fromCache()

        for user_id, projects in data.items():
            #print(type(array(userProjs)))
            result[user_id] = []

            for proj in projects:
                #print(proj)
                joinedText = " ".join([proj["name"], proj["description"]])
                result[user_id].append(joinedText)

        return result