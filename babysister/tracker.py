"""Simple online multiple objects tracker powered by OpenCV
"""
import collections
import cv2 as cv
from pprint import pprint


class NaiveTracker:
    trackers = {
        'BOOSTING': cv.TrackerBoosting_create,
        'MIL': cv.TrackerMIL_create,
        'KCF': cv.TrackerKCF_create,
        'TLD': cv.TrackerTLD_create,
        'MEDIANFLOW': cv.TrackerMedianFlow_create,
        'GOTURN': cv.TrackerGOTURN_create,
        'MOSSE': cv.TrackerMOSSE_create,
        'CSRT': cv.TrackerCSRT_create
    }

    def __init__(self, tracker_name='CSRT', max_disappeared=50):
        if tracker_name not in self.trackers.keys():
            print('Incorrect tracker name')
            print('Available trackers are:')
            print('\n'.join(list(self.trackers.keys())))
        
        # tracker for each object
        self.tracker_name = tracker_name
        
        # all objects that being tracked
        self.next_id = 0
        self.objects = collections.OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, frame, boxes):
        """Register new objects"""
        for box in boxes:
            self.objects[self.next_id] = {
                'tracker': self.trackers[self.tracker_name]().init(frame, box),
                'box': box,
                'disappeared': 0
            }
            self.next_id += 1

    def deregister(self, id_):
        del self.objects[id_]

    def update(self, frame, detected_boxes):
        # update all objects that being tracked
        ids = self.objects.keys()
        for id_ in ids:
            ok, box = self.objects[id_]['tracker'].update(frame)

            if ok:
                self.objects[id_]['box'] = box
                self.objects[id_]['disappeared'] = 0
            else:
                self.objects[id_]['box'] = ()
                self.objects[id_]['disappeared'] += 1
                if self.objects[id_]['disappeared'] > max_disappeared:
                    self.deregister(id_)

        for detected_box in detected_boxes:
            for id_, obj in 




    
    
    

