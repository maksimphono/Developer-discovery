import sys
sys.path.append('/home/trukhinmaksim/src')

from src.utils.DatasetManager import DatasetManager

class NormalizerRemover(DatasetManager):
    # will normalize the dataset by removing tags from tag list of each document, tags that must be deleted are deleted first, then other tags are removed from the end while number of tags are bigger than 6
    KEEP_TAGS_AMOUNT = 12

    def __init__(self, tagsToKeep = [], tagsDistribution = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tagsToKeep = set(tagsToKeep)
        self.tagsDistribution = tagsDistribution

    def mapper(self, doc):
        for tag in (set(doc["tags"][2:]) - self.tagsToKeep):
            if len(doc["tags"]) <= NormalizerRemover.KEEP_TAGS_AMOUNT: break
            doc["tags"].remove(tag)

        # if after I've deleted some tags, number of remaining tags are too big, keep removing most unpopular tags
        i = 0
        while len(doc["tags"]) > NormalizerRemover.KEEP_TAGS_AMOUNT:
            doc["tags"].pop()

        return doc