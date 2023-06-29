import json
import os


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