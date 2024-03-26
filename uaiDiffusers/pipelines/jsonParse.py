import json
import os
from uaiDiffusers.media.mediaRequestBase import MediaRequestBase64, MultipleMediaRequest


def DictLoadValue(key, dictData, defaultValue):
    """
    Load a value from a dict if it exists, otherwise return a default value
    
    Args:
        key (str): The key to look for in the dict
        dictData (dict): The dict to look in
        defaultValue (any): The value to return if the key is not found
    
    Returns:
        any (object): The value of the key if it exists, otherwise the default value
    """
    try:
        if key in dictData:
            return dictData[key]
        else:
            return defaultValue
    except:
        return defaultValue
    
def JsonStringLoadValue(key, jsonString, defaultValue):
    """
    Load a value from a json string if it exists, otherwise return a default value
    
    Args:
        key (str): The key to look for in the dict
        jsonString (str): The json string to load and search
        defaultValue (any): The value to return if the key is not found
    
    Returns:
        any (object): The value of the key if it exists, otherwise the default value
    """
    try:
        jsonData = json.loads(jsonString)
        if key in jsonData:
            return jsonData[key]
        else:
            return defaultValue
    except:
        return defaultValue
    
    
import base64, io, json
from flask import jsonify
import PIL.Image


def ImageToBytes(pilImage):
    '''
    Takes a PIL image and convert it to a BytesIO object
    
    Args:
        pilImage (PIL.Image): PIL image
    
    Returns:
        img_io (BytesIO): BytesIO object
    '''
    img_io = io.BytesIO()
    pilImage.save(img_io, 'PNG', quality=100)
    img_io.seek(0)
    # return the pilImage as an image
    return img_io


def ImagesToBase64(pilImage):
    '''
    
    Takes a PIL image and convert it to a Base64 string
    
    Args:
        pilImage (PIL.Image): PIL image
    
    Returns:
        base64String (str): Base64 string
    '''
    # return the pilImage as an image
    return  base64.b64encode(ImageToBytes(pilImage).getvalue()).decode()


def MultiImagesToFlaskResponse(pilImages, prompts):
    '''
    Takes a list of PIL images and prompts of the same count and returns a MultipleMediaRequest JSON object containing the images and prompts as MediaRequestBase64 objects
    
    Args:
        pilImages (list): A list of PIL images
        prompts (list): A list of prompts
        
    Returns:
        JSONString (str): A MultipleMediaRequest JSON string
    '''
    # return the pilImage as an image
    
    MultipleMediaRequest_ = MultipleMediaRequest([])
    for indx, img in enumerate(pilImages):
        prompt = ""
        try:
            prompt = prompts[indx]
        except:
            prompt = ""
        MultipleMediaRequest_.addMedia(MediaRequestBase64(ImagesToBase64(img), prompt))
        
    return jsonify(MultipleMediaRequest_.toDict())

def MultiImagesToJSONResponse(pilImages, prompts):
    '''
    Takes a list of PIL images and prompts of the same count and returns a MultipleMediaRequest JSON object containing the images and prompts as MediaRequestBase64 objects
    
    Args:
        pilImages (list): A list of PIL images
        prompts (list): A list of prompts
        
    Returns:
        JSONString (str): A MultipleMediaRequest JSON string
    '''
    # return the pilImage as an image
    
    MultipleMediaRequest_ = MultipleMediaRequest([])
    for indx, img in enumerate(pilImages):
        prompt = ""
        try:
            prompt = prompts[indx]
        except:
            prompt = ""
        MultipleMediaRequest_.addMedia(MediaRequestBase64(ImagesToBase64(img), prompt))
        
    return json.dumps(MultipleMediaRequest_.toDict())

def Base64StringToPILImage(base64String):
    '''
    Takes a base64 string and returns a PIL image
    '''
    try:
        return PIL.Image.open(io.BytesIO(base64.b64decode(base64String)))
    except:
        return None

def uaiPromptImageJSONObjectToPILImage(promptImage):
    '''
    Takes a prompt image from the UAI API and returns a PIL image
    
    Args:
        promptImage (dict): A prompt image from the UAI API
    
    Returns:
        PIL.Image: A PIL image
    '''
    return Base64StringToPILImage(promptImage["media"])


def uaiPromptMultiImageJSONObjectToPILImage(promptImages):
    '''
    Takes a prompt image from the UAI API and returns a PIL image
    '''
    imgs = []
    for indx, img in enumerate(promptImages['media']):
        imgs.append(Base64StringToPILImage(img["media"]))
    return imgs

def saveUAIPromptMultiPILImageObjectToFiles(promptImages, filepath):
    for indx , img in enumerate(promptImages):
        img.save(filepath + str(indx).zfill(4) + ".png")


def saveUAIPromptImageJSONObjectToFiles(promptImage, filepath):
    '''
    Save the image to file
    
    Args:
        promptImage (dict): A prompt image from the UAI API
        filepath (str): The path to save the image to
    
    Returns:
        PIL.Image: A PIL image
    '''
    piImage = Base64StringToPILImage(promptImage["media"])
    piImage.save(filepath)
    return piImage


def saveUAIPromptMultiImageJSONObjectToFiles(promptImages, filepath):
    for indx , img in enumerate(promptImages['media']):
        saveUAIPromptImageJSONObjectToFiles(img, filepath  + str(indx).zfill(4) + ".png")