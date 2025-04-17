from pymongo import MongoClient

# single machine setup (mongo is running here localy)
# "ip a" for ip address
MY_DATABASE_LINK = 'mongodb://10.22.112.39:27020/' #'mongodb://192.168.100.57:27020/'
# multiple mechine setup (mongo is running on another machine)
#MY_DATABASE_LINK = 'mongodb://192.168.43.78:27020/'

class DatabaseConnect:
    DB_LINK = MY_DATABASE_LINK

    class Base:
        client = None
        @classmethod
        def connect(cls, databaseName):
            cls.client = MongoClient(DatabaseConnect.DB_LINK)
            # Access the database
            return cls.client[databaseName]

        @classmethod
        def close(cls):
            if cls.client:
                cls.client.close()
                cls.client = None

        @classmethod
        def getCollection(cls, collectionName):
            return cls.client[collectionName]

    class mini_database(Base):
        @classmethod
        def projects(cls):
            #print(cls.connect)

            return cls.connect('mini_database')['projects']
        @classmethod
        def users(cls):
            return cls.connect('mini_database')['users']

        @classmethod
        def cache(cls):
            return cls.connect('mini_database')['cache']

    class developer_discovery(Base):
        @classmethod
        def proj_info(cls):
            return cls.connect('developer_discovery')['proj_info']

class WL_DatabaseConnect(DatabaseConnect):
    class mini_database:
        pass


class DatabaseConnector:
    def __init__(self, link, dbName):
        self.client = MongoClient(link)
        self.dbName = dbName

    def connect(self):
        return self.client[self.dbName]

    def collection(self, name):
        return self.connect()[name]

class CacheConnector(DatabaseConnector):
    def __init__(self, link):
        super().__init__(link, "Cache")

class CacheConnector_02_04_25(CacheConnector):
    @property
    def train_02_04_25(self):
        return self.collection("train_02-04-25")

    @property
    def test_02_04_25(self):
        return self.collection("test_02-04-25")