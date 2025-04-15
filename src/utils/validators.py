# Validators are used to filter data by quality, 
# for example, I can take only those project, that has long description, readme file and many stars


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