import numpy as np

def flatternData(data : dict[str, list]) -> np.array(dict):
    # takes in data in form of dict, where each key is a user id and each value is a list of that user's projects
    # returns just flat list of these projects 
    result = []

    for projectsArray in data.values():
        for project in projectsArray:
            result.append(project)

    return result

