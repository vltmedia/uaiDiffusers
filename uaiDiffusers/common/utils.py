import os
CLRMaxSteps = 20


def tempSetResponse(stt):
    print("")
def tempResponse():
    print("")

def tempProgress(percent, message, isDone):
    print("")

def tempProgressResponse(steps, timescale, latents):
    print("")

def tempImageProgressResponse(diffusers, steps, timescale, latents):
    return latents

def SetCLRMaxSteps(steps):
    global CLRMaxSteps
    CLRMaxSteps = steps

def GetCLRMaxSteps():
    global CLRMaxSteps
    return CLRMaxSteps

def GetAIModelsCachePath():
    return os.environ.get("DIFFUSERS_CACHE",os.path.expanduser("~") + "/.cache/huggingface/hub")

class UAICallBacks:
    def __init__(self,  sendProgress = tempProgress, onFinishedCallback = tempResponse, progressCallback = tempProgressResponse, imageProgressCallback = tempImageProgressResponse, setResponseMessageCallback = tempSetResponse):
        self.sendProgress = sendProgress
        self.onFinishedCallback = onFinishedCallback
        self.progressCallback = progressCallback
        self.imageProgressCallback = imageProgressCallback
        self.setResponseMessage = setResponseMessageCallback
        
    def sendProgress(self, percent, message, isDone):
        self.sendProgress(percent, message, isDone)
    
    def onFinishedCallback(self):
        self.onFinishedCallback()
        
    def progressCallback(self, steps, timescale, latents):
        return self.progressCallback(steps, timescale, latents)
    
    def imageProgressCallback(self, diffusers, steps, timescale, latents):
        return self.imageProgressCallback(diffusers, steps, timescale, latents)
    
    def setSendProgress(self, sendProgress):
        self.sendProgress = sendProgress
        
    def setOnFinishedCallback(self, onFinishedCallback):
        self.onFinishedCallback = onFinishedCallback
    
    def setProgressCallback(self, progressCallback):
        self.progressCallback = progressCallback
        
    def setImageProgressCallback(self, imageProgressCallback):
        self.imageProgressCallback = imageProgressCallback
        
    def setResponseMessageCallback(self, setResponseMessageCallback):
        self.setResponseMessage = setResponseMessageCallback
        
    def getSendProgress(self):
        return self.sendProgress
    
    def getOnFinishedCallback(self):
        return self.onFinishedCallback
    
    def getProgressCallback(self):
        return self.progressCallback
    
    def getImageProgressCallback(self):
        return self.imageProgressCallback
    
        
