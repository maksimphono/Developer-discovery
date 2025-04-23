import sys
sys.path.append('/home/trukhinmaksim/src')

from src.utils.DatasetManager import DatasetManager

class NormalizerRemover(DatasetManager):
    # will normalize the dataset by removing tags from tag list of each document, tags that must be deleted are deleted first, then other tags are removed from the end while number of tags are bigger than 6
    def __init__(self, tagsToRemove = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tagsToRemove = set(tagsToRemove)

    def mapper(self, doc):
        for tag in set(doc["tags"]) & self.tagsToRemove:
            if len(doc["tags"]) <= 6: break
            doc["tags"].remove(tag)

        # if after I've deleted some tags, number of remaining tags are too big, keep removing from the end
        while len(doc["tags"]) > 6:
            doc["tags"].pop()

        return doc