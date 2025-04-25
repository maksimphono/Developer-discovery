import sys
sys.path.append('/home/trukhinmaksim/src')

from src.utils.DatasetManager import DatasetManager

class NormalizerRemover(DatasetManager):
    # will normalize the dataset by removing tags from tag list of each document, tags that must be deleted are deleted first, then most unpopular tags are removed untill the list is too long
    KEEP_TAGS_AMOUNT = 6
    ITEMS_PORION_LEN = 1000

    def __init__(self, tagsToKeep = list(), tagsDistribution = dict(), *args, **kwargs):
        super().__init__(NormalizerRemover.ITEMS_PORION_LEN, *args, **kwargs)
        self.tagsToKeep = set(tagsToKeep)
        self.tagsDistribution = tagsDistribution
        self.mapper = self.normalize

    def normalize(self, doc):
        tags = doc["tags"][2:] # not counting id and name

        for tag in (set(tags) - self.tagsToKeep):
            if len(tags) <= NormalizerRemover.KEEP_TAGS_AMOUNT: break
            tags.remove(tag)

        # if after I've deleted some tags, number of remaining tags are too big, keep removing most unpopular tags
        if len(tags) > NormalizerRemover.KEEP_TAGS_AMOUNT:
            removeTagsAmount = len(tags) - NormalizerRemover.KEEP_TAGS_AMOUNT # how much tags is still there to remove

            tagsToRemove = (tag[0] for tag in self.selectKmostUnpopular(tags, removeTagsAmount))

            for tag in tagsToRemove:
                tags.remove(tag)

        doc["tags"][2:] = tags
        return doc


    def selectKmostUnpopular(self, tags, k):
        selected = [("", float("inf"))] # works like monotonic stack, projects with higher score are pushed higher (closer to the en

        def insert(tag, score):
            nonlocal selected, k
            inserted = False

            # starting from 1 because that function is called only if score is higher then the 0-th element, so here isn't necessary to check it again
            for i, pair in enumerate(selected[1:], 1):
                if score >= pair[1]:
                    selected.insert(i, (tag, score))
                    inserted = True
                    break

            if not inserted:
                # if the new score is the highest
                selected.append((tag, score))

            if len(selected) > k:
                selected.pop(0)


        for tag in tags:
            # traverse through all vectors, here vectors are listed in the same order as in the corpus, so I'm recoring index of each vector
            score = self.tagsDistribution[tag]

            if score < selected[0][1]: # if the insertation is needed in the first place
                insert(tag, score)

        return selected