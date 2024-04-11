import requests
import json, os
from uaiDiffusers.uaiDiffusers import AddCreateJob, GetAUserStrapiID,AddNewMediaContent
from authovalidator.validator import Auth0JWTBearerTokenValidator
from authlib.integrations.flask_oauth2 import ResourceProtector

def SetupFlaskAuth0():
    """
    Setup the Auth0 Authentication for the Flask App. Uses the Auth0 Domain and Audience from the environment variables AUTH0_DOMAIN and AUTH0_AUDIENCE
    
    Returns:
        ResourceProtector: The Auth0 ResourceProtector for the Flask App.
    """
    require_auth = ResourceProtector()
    validator = Auth0JWTBearerTokenValidator(
    os.environ.get("AUTH0_DOMAIN"),
    os.environ.get("AUTH0_AUDIENCE")
)
    require_auth.register_token_validator(validator)
    return require_auth
    
def UploadMedia(media, path , mimeType = "image/jpeg"):
    """
    Upload media to the server
    
    Args:
        media (dict): The media to upload as {"media":[]}
    
    Returns:
        jobId (str): The JobId of the job created by the server
    """
    outreq = {"path":path,"media":media, "mimeType":mimeType}
    request = requests.post("https://faas-nyc1-2ef2e6cc.doserverless.co/api/v1/web/fn-33d7e743-bdcf-4b28-8afd-5589d75e6700/uaibackend/uaibackenduploadmedia", json=outreq)
    open("upload.json", "w").write(json.dumps(outreq))
    print(request.reason)
    print(request.status_code)
    return request.json()

def GenerateFilePath(user, jobID, index = 0, extension = "jpg"):
    """
    Generate the filepath for a given User
    
    Args:
        user (str): The user to generate the filepath for
        jobID (str): The jobID to generate the filepath for
        index (int): The index of the file to generate the filepath for
        extension (str): The extension of the file to generate the filepath for
    
    Returns:
        str: The filepath for the given user, jobID, index, and extension
    """
    return str.format("users/{0}/{1}/render/{1}_{2}.{3}", user,jobID, str(index).zfill(4),extension)

def SetJobStatus(job, status = "waiting"):
    """
    Set the status of a job
    
    Args:
        job (dict): The job to set the status of
        status (str): The status to set the job to
    
    """
    job['status'] = status
    request = requests.post("https://faas-nyc1-2ef2e6cc.doserverless.co/api/v1/web/fn-33d7e743-bdcf-4b28-8afd-5589d75e6700/uaibackend/uaibackendupdatejob", json={"job":job})
    

def SetJobToFinished(job):
    
    """
    Set the status of a job to finished
    
    Args:
        job (dict): The job to set the status of
    
    """
    SetJobStatus(job,"finished")
    


def GenerateImage(job, userID):
    """
    Generate a SD Image from a job and UserID
    
    Args:
        job (dict): The job to generate the image from
        userID (str): The UserID to generate the image for
    """
    try:
        if isinstance(job['job']['output'] , list) == False:
            job['job']['output'] = []
    except:
        p = 0
    print("Updating Job to Processing")
    SetJobStatus(job['job'], "processing")
    
    url = 'http://localhost:6678'
    request = requests.post(url + job['job']['route'], json=json.loads(job['job']['data']))

    media = request.json()['media']
    for index, media_ in enumerate( media):
        url_ = GenerateFilePath(job['job']['user'].replace("|","-"), job['job']['jobID'], index = index, extension = "jpg") 
        newMedia = AddNewMediaContent(url_, job['job']['route'], job['job']['data'],userID)
        job['job']['output'].append(
            newMedia['id']
        )   
        print(job['job']['output'])
        print("Uploading Media to " + url_)
        UploadMedia( media_['media'],url_, "image/jpeg")
        
    SetJobToFinished(job['job'])
    


def RouteJob(job, userID):
    """
    Run the appropriate function for a given job
    
    Args:
        job (dict): The job to run the function for
        userID (str): The UserID to run the function for
        
    """
    jobRoute = job['route']
    
    if jobRoute == "/image":
        GenerateImage({"job":job}, userID)
    # Start Job
    # if jobRoute == "audio/say/custom/tts/vits":
    #     getTTSVitsCustomAudioGet(jobData['data'])
    # elif jobRoute == "fortune":
    #     fortune()
    # elif jobRoute == "image":
    #     image(jobData['data'])
    # else:
    #     requestPassthrough
    # return jobData