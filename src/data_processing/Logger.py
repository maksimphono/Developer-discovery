import os

class Logger:
    logDirectory = ""
    def __init__(self, fileName):
        self.logs = []
        self.fileName = "logs.txt" 

    def flash(self):
        for log in self.logs:
            with open(os.path.join(self.logDirectory, log.file), "a+") as file:
                file.write(log.content)

    def logScannedCSVFile(self, )