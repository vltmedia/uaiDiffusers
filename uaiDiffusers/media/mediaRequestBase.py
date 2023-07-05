import json
class MediaRequestBase64:
    """
    Class to hold the media as a base64 string along with the prompt used to generate it
    """
    def __init__(self, media,prompt):
        """
        Args:
            media (str): The base64 string of the media
            prompt (str): The prompt used to generate the media
        """
        self.media = media
        self.prompt = prompt
        
        
class MultipleMediaRequest:
    """
    Class used to hold multiple MediaRequestBase64 objects and return them if needed
    """
    def __init__(self, media = []):
        """
        Args:
            media (list): A list of MediaRequestBase64 objects
        """
        self.media = media
        
    def addMedia(self, media):
        """
        Add a MediaRequestBase64 object to the list of media
        
        Args:
            media (MediaRequestBase64): The media to add
        """
        self.media.append(media)
        
    def toDict(self):
        """
        Convert the object to a json string
        
        Returns:
            str: The json string
        """
        media = []
        for m in self.media:
            media.append(m.__dict__)
        return {"media":media}
        
    def toJson(self):
        """
        Convert the object to a json string
        
        Returns:
            str: The json string
        """
        return json.dumps(self, default=lambda o: o.__dict__)