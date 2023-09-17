import bpy
from uaiDiffusers.scene.scene import *
    
def GetTrackers():
    """
    Returns all of the current tracker markers in a Blender VFX Movie Clip scene as a UAIVFXTrackedScene object
    """
    D = bpy.data
    outputScene = UAIVFXTrackedScene(bpy.path.basename(bpy.context.blend_data.filepath).split('.')[0])
    printFrameNums = False # include frame numbers in the csv file
    relativeCoords = False # marker coords will be relative to the dimensions of the clip
    for clip in D.movieclips:
        print('clip {0} found\n'.format(clip.name))
        width=clip.size[0]
        height=clip.size[1]
        for ob in clip.tracking.objects:
            print('object {0} found\n'.format(ob.name))
                
            for track in ob.tracks:
                tracker = Tracker(track.name)
                print('track {0} found\n'.format(track.name))
                framenum = 0
                while framenum < clip.frame_duration:
                    markerAtFrame = track.markers.find_frame(framenum)
                    if markerAtFrame:
                        coords = markerAtFrame.co.xy
                        newKey = Keyframe(framenum, Transform2D(position = [coords[0], coords[1]]))
                        if relativeCoords:
                            newKey.transform.SetPositionXY(coords[0], coords[1])
                        else:
                            newKey.transform.SetPositionXY(coords[0]*width, coords[1]*height)
                        tracker.keyframes.append(newKey)
                    framenum += 1
                outputScene.trackers.append(tracker)
    return outputScene

def ExportTrackers(filePath):
    uaiScene = GetTrackers()
    outData = uaiScene.json()
    open(bpy.path.abspath( filePath), 'w').write(outData)