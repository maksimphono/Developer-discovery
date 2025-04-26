import numpy as np
from time import time

def flatternData(data : dict[str, list]) -> np.array(dict):
    # takes in data in form of dict, where each key is a user id and each value is a list of that user's projects
    # returns just flat list of these projects 
    result = []

    for projectsArray in data.values():
        for project in projectsArray:
            result.append(project)

    return result


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def getTagsQuantitiesForCorpus(corpus):
    tagsCount = {}
    tagsLst = []

    def sort(tagsCount):
        print("Sorting tags")
        tagsLst = sorted([*tagsCount.items()], key = lambda pair_1: pair_1[1], reverse = True)

        return tagsLst


    start = time()
    i = 0
    try:
        for proj in corpus:
            if i % 100000 == 0:
                print(f"Scanned {i} projects in {time() - start} s")
            for tag in proj.tags:
                if tag in tagsCount:
                    tagsCount[tag] += 1
                else:
                    tagsCount[tag] = 1

            i += 1
    except Exception as exp:
        raise exp
    finally:
        tagsLst = sort(tagsCount)
        tagsCount.clear()
    
        return tagsLst