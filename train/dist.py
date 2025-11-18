import os

def is_master():
    if int(os.environ["RANK"]) == 0:
        return True
    else:
        return False