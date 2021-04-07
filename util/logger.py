import logging
import time
import sys
import os
import datetime


def logger():
    """
    usage: ```
    log = logger()
    log.debug("42") # <-- print '42' on console and in the file in folder './log'
    ```
    :return: Logger instance
    """
    logFormatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    rootLogger = logging.getLogger()
    rootLogger.setLevel(10)

    filename = f"{datetime.datetime.now().isoformat().replace(':', '-')}"
    logPath = "{0}/{1}.log".format("log", filename)
    if not os.path.exists("log"):
        os.mkdir("log")

    fileHandler = logging.FileHandler(logPath)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger