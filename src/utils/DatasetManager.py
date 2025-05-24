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
from langdetect.lang_detect_exception import LangDetectException
import requests
from json import dumps
from random import choice
import textstat
import asyncio
import nest_asyncio
from googletrans import Translator
import traceback
import fasttext
import logging
import markdown
from bs4 import BeautifulSoup

from src.utils.CacheAdapter import JSONAdapter, CacheAdapter, EXP_END_OF_DATA
from src.data_processing.collect_projects_data import collectOneProjectData, EXP_NOT_IN_DB
from src.utils.DatabaseConnect import DatabaseConnector

def downloadArgosLangPackages(langList = ["es", "pt", "zh", "zt", "ru", "de", "ja", "ko"]):
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    packages = list(filter(lambda pkg: pkg.from_code in langList and pkg.to_code == "en", available_packages))
    print(packages)
    for pkg in packages:
        argostranslate.package.install_from_path(pkg.download())

try:
    fastTextModel = fasttext.load_model('/home/trukhinmaksim/lid.176.bin')
except ValueError as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'lid.176.bin' is in the current directory or provide the full path.")
    print("You can download it from: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
    exit()

def detectLang(text):
    global fastTextModel
    predictions = fastTextModel.predict(text)
    languageLabel = predictions[0][0]
    languageCode = languageLabel.replace('__label__', '')

    return languageCode

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
        self.data = {}
        self.preprocessed = False
        self.ignoredUsers = IgnoreList()
        self.readability = {"flesch" : 13, "dale_chall" : 11}
        self.readabilityCheckEnabled = True
        self.stop_words = set(stopwords.words("english") + ["etc"])
        self.lemmatizer = WordNetLemmatizer()

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
                    # if user has at least one project he contributed to, that passed validation check
                    print(f"Scanning {user['id']}")
                    data[user["id"]] = deepcopy(projects)
                    count -= 1

                i += 1

        return data

    def translateText(self, text, retry = 3, useServer = False):
        # will try to use Google Translate, but if any error occures, will use Argos offline translator
        try:
            if text.isascii() or detect(text) == "en": return text # if the text is already english (either ascii or english with unicode emoji)
        except LangDetectException:
            pass

        if useServer:
            translatorURL = choice(ProjectsDatasetManager.translatorServers)

            #print(translatorURL)
            response = requests.request("POST", url = translatorURL, headers = {'Content-Type': 'application/json'}, data = dumps({"text" : text}, ensure_ascii=False, indent=4))

            if response.ok:
                return response.json()["translate"]
        else:
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
                    # retry translation
                    sleep(random() * 6)
                    print("Conection error, retrying")
                    continue

        # failed translation, trying to use argos translator
        langCode = detect(text)[:2]
        if langCode not in ["es", "pt", "zh", "zt", "ru", "de", "ja", "ko"]: langCode = "zh" # If language is unknown, assuming it is Chinese
        print(f"Translation Faled after {retry} attempts, Using Argos for {text[:10]}...\nLanguage detected as: {langCode}")
        return argostranslate.translate.translate(text, langCode, "en")


    def checkTextReadability(self, text):
        return (textstat.flesch_reading_ease(text) >= self.readability["flesch"] and textstat.dale_chall_readability_score(text) >= self.readability["dale_chall"])

    def textPreprocessing(self, text):
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
        text = re.sub(r"\d", "", text)
        # Remove new lines:
        text = re.sub(r"\n", " ", text)
        # Remove multiple spaces:
        text = re.sub("\s+", " ", text).strip()

        tokens = [word for word in word_tokenize(text) if word not in self.stop_words and len(word) > 1]  # Tokenize into words

        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Remove stopwords & lemmatize

        return tokens

    def projectsDataPreprocessing(self, projects : array(dict), includingText = False) -> array([{"tokens" : str, "tags" : list}]):
        # will take in an array of projects and prepare it to be consumed by the model
        # takes: array of projects (as dictionaries); returns: text data and tags for every project in array
        result = []

        for proj in projects:
            joinedText = " ".join([proj["name"], proj["description"]])

            tokens = self.textPreprocessing(joinedText)
            # only meaningfull tags will be saved, no empty strings!
            tags = list(filter(lambda n: (n != ""), [proj["id"], proj["name"], proj["language"]] + proj["topics"]))# if proj["language"] else proj["topics"]

            if includingText:
                result.append({"text" : joinedText, "tokens" : tokens, "tags" : tags})
            else:
                result.append({"tokens" : tokens, "tags" : tags})
                

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


class DatasetManager:
    # read data from the input (Database or cache), filters it with 'validator', applies 'mapper' and writes into the 'outputAdapters'
    class DatasetManagerError(Exception):
        pass

    EXP_INPUT_ERROR = DatasetManagerError("Input error")
    EXP_OUTPUT_ERROR = DatasetManagerError("Output error")

    def __init__(self, itemsPortionNum, inputAdapter, outputAdapters = list(), limit = float("inf"), validator = lambda x: True, mapper = lambda x: x):
        self.itemsPortionNum = itemsPortionNum # how many items can be in one portion, portion will be written into the outputs once full
        self.limit = limit # how many items can be scanned overall
        self.inputAdapter = inputAdapter # where to take data from
        self.outputAdapters = outputAdapters # where to write data to
        self.validator = validator
        self.mapper = mapper # function, that will be applied to the validated object, must return modified object
        self.data = []
        self.mappedData = []
        self.blackList = None # items, that must be ignored
        self.readCounter = 0
        self.totalScannedProjects = 0

    def ignore(self, doc):
        if self.blackList:
            self.blackList.add(doc)

    def readInput(self):
        if self.readCounter >= self.limit: 
            return []

        data = []
        while len(self.data) < self.itemsPortionNum:
            try:
                data = self.inputAdapter.load(1)
            except EXP_END_OF_DATA:
                if len(self.data):
                    self.readCounter += len(data)
                    return self.data
                else:
                    raise EXP_END_OF_DATA

            if self.blackList:
                if not self.blackList.includes(data[0]):
                    self.data += data#list(filter(lambda doc: self.blackList.contains(doc), data))
                    self.blackList.add(data[0])
            else:
                self.data += data

        self.readCounter += len(self.data)
        print(f"Counter: {self.readCounter}")
        return self.data

    def writeOutput(self):
        for output in self.outputAdapters:
            output.save(self.mappedData)

        self.data.clear()
        self.mappedData.clear()

    def __call__(self):
        self.readInput()

        for doc in self.data:
            if self.validator(doc):
                updatedDoc = self.mapper(doc)
                self.mappedData.append(updatedDoc)

            self.totalScannedProjects += 1

        self.writeOutput()
        


# TODO: change implementation of that class, optimize the performance, increase reusability
class NewDatasetManager(DatasetManager):
    translatorServers = []

    class EXP_CONNECTION_LOSS(Exception):
        def __init__(self):
            super().__init__("Can't establish connection with 'Google translate'")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_words = set(stopwords.words("english") + ["etc"])
        self.lemmatizer = WordNetLemmatizer()
        self.mapper = self.preprocess
        self.processedProjsIds = []
        self.processedProjectsNum = 0

    def translateText(self, text, retry = 3, useServer = False, precheck = True):
        # will try to use Google Translate, but if any error occures, will use Argos offline translator
        try:
            if precheck and (text.isascii() or detectLang(text.replace("\n", "")) == "en"): 
                return text # if the text is already english (either ascii or english with unicode emoji)
        except LangDetectException:
            pass

        if useServer:
            translatorURL = choice(NewDatasetManager.translatorServers)

            #print(translatorURL)
            #print(f"Connecting through server, text = '{text}', got response:")
            response = requests.request("POST", url = translatorURL, headers = {'Content-Type': 'application/json'}, data = dumps({"text" : text}, ensure_ascii=False, indent=4))

            print(response)
            if response.ok:
                return response.json()["translate"]
            else:
                raise NewDatasetManager.EXP_CONNECTION_LOSS
        else:
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
                    # retry translation
                    sleep(random() * 6)
                    logging.info("Conection error, retrying")
                    continue

        # failed translation, trying to use argos translator
        raise NewDatasetManager.EXP_CONNECTION_LOSS()
        #langCode = detect(text)[:2]
        #if langCode not in ["es", "pt", "zh", "zt", "ru", "de", "ja", "ko"]: langCode = "zh" # If language is unknown, assuming it is Chinese
        #print(f"Translation Faled after {retry} attempts, Using Argos for {text[:10]}...\nLanguage detected as: {langCode}")
        #return argostranslate.translate.translate(text, langCode, "en")

    def textPreprocessing(self, text, translate = True):
        # Translate:
        if translate: 
            text = self.translateText(text, 3, False)
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
        text = re.sub(r"\d", "", text)
        # Remove new lines:
        text = re.sub(r"\n", " ", text)
        # Remove multiple spaces:
        text = re.sub("\s+", " ", text).strip()

        tokens = [word for word in word_tokenize(text) if word not in self.stop_words and len(word) > 1]  # Tokenize into words

        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Remove stopwords & lemmatize

        return tokens

    def projectDataPreprocessing(self, project : array(dict), includingText = False) -> array([{"tokens" : str, "tags" : list}]):
        # will take in an array of projects and prepare it to be consumed by the model
        # takes: array of projects (as dictionaries); returns: text data and tags for every project in array
        result = []

        joinedText = " ".join([project["name"], project["description"]])
        #joinedText = ""
        #if project["name"]:
        #    joinedText += project["name"]
        #if project["description"]:
        #    joinedText += " " + project["description"]

        tokens = self.textPreprocessing(joinedText)

        # only meaningfull tags will be saved, no empty strings!
        tags = list(filter(lambda n: (n != ""), [project["proj_id"], project["name"], project["language"]] + project["topics"]))# if proj["language"] else proj["topics"]
        if includingText:
            result = {"text" : joinedText, "tokens" : tokens, "tags" : tags}
        else:
            result = {"tokens" : tokens, "tags" : tags}

        return result

    def handleConnectionLoss(self, payload = {}):
        print(f"Can't connect to 'Google translate', fix internet connection and try again")
        logging.error(f"Can't connect to 'Google translate', fix internet connection and try again; {payload}")
        command = input()
        if command == "c":
            return True
        else:
            return False

    def handleException(self):
        message = traceback.format_exc()
        print(f"Encountered exception, content:\n{message}\nFix it and type 'c' to proceed!")
        logging.error(f"Encountered exception, content:\n{message}\nFix it and type 'c' to proceed!")
        command = input()
        if command == "c":
            return True
        else:
            return False

    def interruptGracefully(self, exp, message):
        print(message)
        logging.error(message)
        self.writeOutput()
        raise exp

    def preprocess(self, project):
        while True:
            try:
                result = self.projectDataPreprocessing(project)
                self.processedProjsIds.append(project["id"])
                return result
            except NewDatasetManager.EXP_CONNECTION_LOSS as exp:
                if self.handleConnectionLoss():
                    continue # try again
                else:
                    self.interruptGracefully(f"Falied to fix error :(\nError occured while scanning {project['proj_id']}")
            except Exception as exp:
                if self.handleException():
                    continue
                else:
                    self.interruptGracefully(f"Falied to fix error :(\nError occured while scanning {project['proj_id']}")


    def writeOutput(self):
        if self.blackList:
            for _id in self.processedProjsIds:
                self.blackList.add(_id)

        if len(self.mappedData):
            print(f"Writing {len(self.mappedData)} items")
            logging.info(f"Writing {len(self.mappedData)} items")
            for output in self.outputAdapters:
                output.save(self.mappedData)

        print(f"Processed {self.totalScannedProjects} projects in total")
        logging.info(f"Processed {self.totalScannedProjects} projects in total")
        self.data.clear()
        self.mappedData.clear()
        self.processedProjsIds.clear()


class RawTextDatasetManager(NewDatasetManager):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj_info = DatabaseConnector("mongodb://readonlyUser:cictest123456@114.212.84.247:27017/", "developer_discovery").collection("proj_info")

    def reset(self, skip):
        self.inputAdapter.reset()
        if skip > 0:
            self.inputAdapter.load(skip)

    def prepareData(self, project):
        proj_id = project["tags"][0]
        #print(proj_id)
        projData = self.proj_info.find_one({"proj_id" : proj_id}, projection = {"name" : True, "description" : True})
        joinedText = ". ".join([projData["name"], projData["description"]])
        translated = self.translateText(joinedText)

        return {"text" : translated, "tags" : project["tags"]}

    def preprocess(self, project):
        while True:
            try:
                result = self.prepareData(project)
                return result
            except NewDatasetManager.EXP_CONNECTION_LOSS as exp:
                if self.handleConnectionLoss():
                    continue # try again
                else:
                    self.interruptGracefully(f"Falied to fix error :(\nError occured while scanning {project['tags'][0]}")
            except Exception as exp:
                if self.handleException():
                    continue
                else:
                    self.interruptGracefully(f"Falied to fix error :(\nError occured while scanning {project['tags'][0]}")


class ReadmeFilesTranslatonManager(NewDatasetManager):
    def __init__(self, maxLength = 4000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = self.preprocess
        self.maxLength = maxLength # < 5000 for Google translator
        self.counter = 0

    def removeLinks(self, text):
        pattern = re.compile(
            r'\b(?:https?://|www\.)\S+\b',
            re.IGNORECASE  # Ensures matching regardless of case for http/https/www
        )
        return pattern.sub('', text)

    def removeMarkdown(self, rawMarkdown):
        html = markdown.markdown(rawMarkdown)

        soup = BeautifulSoup(html, "html.parser")

        # remove code blocks and other elements
        for pre in soup.find_all('pre'):
            pre.extract()
        # remove <code> tags
        for code in soup.find_all('code'):
            code.extract()

        text = soup.get_text()

        text = self.removeLinks(text)

        # normalize whitespace
        text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        return text

    def truncate(self, text):
        length = len(text)

        while length > self.maxLength:
            length = text.rfind("\n", 0, length)

        if length <= 5:
            return text[0:self.maxLength] # just cut off the rest of content
        return text[0:length]

    async def preprocess(self, loop, obj):
        newObj = dict(obj)
        readme = self.removeMarkdown(obj["readme"])
        readme = self.truncate(readme)

        translated = await loop.run_in_executor(
            None,  # Use the default ThreadPoolExecutor
            self.translateText,
            readme
        )

        newObj["readme"] = translated

        return newObj

    async def runAsync(self):
        if self.readCounter >= self.limit: return EXP_END_OF_DATA
        try:
            self.readInput()
        except EXP_END_OF_DATA:
            return EXP_END_OF_DATA

        if self.readCounter > self.limit:
            self.data = self.data[0:-(self.readCounter - self.limit)]

        loop = asyncio.get_running_loop()

        # Create a list of asynchronous tasks
        tasks = []
        for doc in self.data:
            if self.validator(doc):
                task = self.preprocess(loop, doc)
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        self.mappedData.extend(results)

        self.writeOutput()

    async def call(self):
        # will execute the entire process of this manager, processing the entire dataset

        while 1:
            result = await self.runAsync()
            if result == EXP_END_OF_DATA:
                break

