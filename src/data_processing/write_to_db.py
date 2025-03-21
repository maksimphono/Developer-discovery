from pymongo import MongoClient


def connectCeleron():
    client = MongoClient(MY_DB_LINK)
    # Access the database
    db = client.mini_database
    print(db)
    # Access the collection
    collectionProjects = db.projects

class CeleronConnect:
    DB_LINK = 'mongodb://192.168.43.78:27020/'

    class Base:
        @classmethod
        def connect(cls, databaseName, collectionName):
            client = MongoClient(CeleronConnect.DB_LINK)
            # Access the database
            db = client[databaseName]
            # Access the collection
            collectionProjects = db[collectionName]

            return collectionProjects

    class mini_database(Base):
        @classmethod
        def projects(cls):
            #print(cls.connect)
            return cls.connect('mini_database', 'projects')
        @classmethod
        def users(cls):
            return cls.connect('mini_database', 'users')

    

def writeManyProjects(data):
    collection = CeleronConnect.mini_database.projects()

    for proj in data:
        collection.insert_one(proj)


def writeOne(doc):
    collection = CeleronConnect.mini_database.projects()
    collection.insert_one(doc)