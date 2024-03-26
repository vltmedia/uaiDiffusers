

class LoadedAIPipelines:
    """
    Handles the states of the loaded AI pipelines.
    Is a static class to handle the pipelines.
    
    """
    pipelines = {}
    currentPipeline = ""
    
    
    @staticmethod
    def setCurrentPipeline(name):
        if name in LoadedAIPipelines.pipelines:
            LoadedAIPipelines.currentPipeline = name
            return True
        return False
    
    @staticmethod
    def getCurrentPipeline():
        if LoadedAIPipelines.currentPipeline == "":
            return None
        return LoadedAIPipelines.getPipeline(LoadedAIPipelines.currentPipeline)
    
    @staticmethod
    def addPipeline(name, pipeline):
        LoadedAIPipelines.pipelines[name] = pipeline
    
    @staticmethod    
    def getPipeline( name):
        if name not in LoadedAIPipelines.pipelines:
            return None
        return LoadedAIPipelines.pipelines[name]
    
    @staticmethod
    def removePipeline( name):
        if name in LoadedAIPipelines.pipelines:
            del LoadedAIPipelines.pipelines[name]
    
    @staticmethod        
    def getPipelines():
        return LoadedAIPipelines.pipelines
    
    @staticmethod
    def getPipelineNames():
        return list(LoadedAIPipelines.pipelines.keys())
        