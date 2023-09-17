import json

class Transform2D:
    def __init__(self, position = [0,0], scale = [1,1], rotation = [0,0,0,1]):
        self.position = position
        self.scale = scale
        self.rotation = rotation
    def SetPosition(self, position):
        self.position = position
    def SetPositionXY(self, x, y):
        self.position = [x,y]
    def SetScale(self, scale):
        self.scale = scale
    def SetRotation(self, rotation):
        self.rotation = rotation
    

class Keyframe:
    def __init__(self, frameNum, transform):
        self.frameNum = frameNum
        self.transform = transform
    


class Frame:
    def __init__(self, frameNum = 0, base64 = "", path = ""):
        self.frameNum = frameNum
        self.base64 = base64
        self.path = path
    

class Tracker:
    def __init__(self,name = "Tracker", keyframes = []):
        self.name = name
        self.keyframes = keyframes
        
class UAIVFXTrackedScene:
    def __init__(self, name = "UAIVFXTrackedScene", trackers = [], frames = [] , mediaPath = ""):
        self.name = name
        self.trackers = trackers
        self.frames = frames
        self.mediaPath = mediaPath
        
        
    def AddTracker(self, tracker):
        self.trackers.append(tracker)
    
    def AddTracker(self, name, keyframes):
        self.trackers.append(Tracker(name, keyframes))
    
    def AddKeyframeToTracker(self, trackerName, keyframe):
        for tracker in self.trackers:
            if tracker.name == trackerName:
                tracker.keyframes.append(keyframe)
                return

    def json(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
        
    def load(self, jsonStr):
        self.__dict__ = json.loads(jsonStr)
    
    def save(self, path):
        with open(path, 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.__dict__, 
                )
    
    def loadFromFile(self, path):
        data = open(path).read()
        self.load(data)
        
    
