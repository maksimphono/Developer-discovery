import numpy as np
from time import time
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


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

def cosineSimilarity(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)  # Reshape to (1, n_features)
    vec2 = np.array(vec2).reshape(1, -1)  # Reshape to (1, n_features)

    # Check if the vectors have the same dimension
    if vec1.shape[1] != vec2.shape[1]:
        raise ValueError("Vectors must have the same dimension.")

    # Calculate cosine similarity using scikit-learn
    similarity_matrix = cosine_similarity(vec1, vec2)
    return similarity_matrix[0, 0] # Extract the scalar value

def eucledianDistance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.shape[0] != vec2.shape[0]:
        raise ValueError("Vectors must have the same dimension.")

    similarity_matrix = pairwise_distances([vec1, vec2])
    return similarity_matrix[0, 1] # Extract the scalar value