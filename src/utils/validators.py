def projectDataIsSufficient(projectData):
    # filters sufficient data (has description and one(or both) of topics or language)
    try:
        return (projectData and projectData["description"] and (len(projectData["topics"]) or projectData["language"]))
    except KeyError:
        return False
