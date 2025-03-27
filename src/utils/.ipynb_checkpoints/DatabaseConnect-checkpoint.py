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