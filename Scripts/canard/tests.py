import video_utils
import numpy
import pdb

reader = video_utils.VideoReader('/home/davidgj/drente/projects_v2/009-APR-20-2-90.MOV', 1)
succes, img = reader.read()
pdb.set_trace()
print "End"
