# Validators are used to filter data by quality, 
# for example, I can take only those project, that has long description, readme file and many stars
from math import ceil
from langdetect import detect
import re

def projectDataIsSufficient(projectData):
    # filters sufficient data (has description and one(or both) of topics or language)
    try:
        return (projectData and projectData["description"] and (len(projectData["topics"]) or projectData["language"]))
    except KeyError:
        return False

def projectDataIsGood(projectData):
    # filters good data (has description and both topics and language, at least 2 spaces)
    try:
        return all((
            projectData,
            projectData["description"].count(" ") >= 2, # at least 2 spaces (hoping to find at least 3 words in the description)
            (len(projectData["topics"]) or projectData["language"])
        ))
    except KeyError:
        return False


def processKO(sym):
    code = ord(sym)
    if 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
        return "ja"
    # 韩语谚文
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
        return "ko"
    # 中文（CJK 统一汉字）
    if 0x4E00 <= code <= 0x9FFF:
        return "zh"
    if 0x0E00 <= code <= 0x0E7F:
        return "th"

    return "ko"


def mydetect(text):
    try:
        lang = detect(text)[:2]
        if lang == "ko":
            # special processing for language, that was detected as Korean
            return processKO(text[0])    
        else:
            return lang
    except lang_detect_exception.LangDetectException:
        return "" # failed detect a language

def projectDataIsHighQuality(projectData):
    # filters good data (has description and both topics and language, at least 15 spaces)
    # spaces threshold : 15
    try:
        if not all((
            projectData,
            projectData["name"],
            projectData["fork"] == False,
            (len(projectData["topics"]) and projectData["language"])
        )): return False

        thresholdsSym = {
            "zh" : 23,
            "th" : 90,
            "ja" : 40
        }
        thresholdsSp = { # minimum amount of space symbols
            "ko" : 13,
            "hi" : 21
        }

        threshold = 15
        description = projectData["description"]
        lang = mydetect(description)

        if lang in thresholdsSym:
            return thresholdsSym[lang] <= len(re.sub(r"[\s,.!。，?\(\)（）]", "", description))
        elif lang in thresholdsSp:
            return thresholdsSp[lang] <= description.count(" ")
        else:
            return description.count(" ") >= threshold
        #projectData["description"].count(" ") >= 13, # at least 13 spaces (hoping to find at least 14 words in the description)
    except KeyError:
        return False
    except TypeError:
        return False