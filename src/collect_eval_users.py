#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/home/trukhinmaksim/src')
from random import sample

from src.utils.DatasetManager import DatasetManager#, NewDatasetManager, RawTextDatasetManager
from src.utils.CacheAdapter import EXP_END_OF_DATA, FlatAdapter
from src.data_processing.scan_csv_files import UsersEvaluationAdapter
from src.utils.validators import projectDataIsHighQuality, userProjectsIsHighQuality

PRIORITY_CSV_FILES = [
    "user_profiles_github_23mf_react-native-translucent-modal.csv",
    "user_profiles_github_2017398956_react-native-textinput-maxlength-fixed.csv",
    "user_profiles_github_a7ul_react-native-exception-handler.csv",
    "user_profiles_github_afollestad_material-dialogs.csv"
]

inputAdapter = UsersEvaluationAdapter(PRIORITY_CSV_FILES)

outputCache = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/users_evaluation_21-04-25")

counter = 0
def count(user):
    global counter
    counter += 1
    return user

manager = DatasetManager(
    1,
    inputAdapter = inputAdapter, 
    outputAdapters = [outputCache],
    validator = userProjectsIsHighQuality,
    mapper = count
)

def checkResult(adapter):
    counter = 0
    adapter.reset()

    while True:
        try:
            adapter.load(1)
            counter += 1
        except EXP_END_OF_DATA:
            print(f"{counter} items in adapter")

def main():
    print("Welcome!")

    for i in range(10):
        try:
            manager()
            print(f"Collected {counter} high quality users in {i + 1} iterations")
        except EXP_END_OF_DATA:
            break

    checkResult(outputCache)


if __name__ == "__main__":
    main()
