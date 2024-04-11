import os
SERVERMODE = False

def GetDiffusersCache():
    path = os.environ.get("DIFFUSERS_CACHE",os.path.expanduser("~") + "/.cache/huggingface/hub")
    if path == "":
        path =  os.path.expanduser( '~' ) + "/.cache/huggingface/diffusers"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def GetDiffusersCachePath(path):
    newPath = GetDiffusersCache() + "/" + path
    if not os.path.exists(newPath):
        os.makedirs(newPath)
    return newPath

def GetServerMode():
    return SERVERMODE

def SetServerMode(mode):
    global SERVERMODE
    SERVERMODE = mode