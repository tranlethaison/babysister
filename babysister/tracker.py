"""Wraper for objects tracking algorithms
cite:
"""
from pprint import pprint
from .utils import iou


class IOUtracker():
    """Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.
    """
    def __init__(self, sigma_l=0, sigma_h=0.5, sigma_iou=0.5, t_min=2):
        """
        Args:
            sigma_l (float): low detection threshold.
            sigma_h (float): high detection threshold.
            sigma_iou (float): IOU threshold.
            t_min (float): minimum track length in frames.
        """
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min

        self.tracks_active = []
        self.tracks_finished = []

    def gen_detections(self, boxes, scores):
        """Generate detections to use with IOU based tracker
        [
            {'bbox': (x1,y1,x2,y2), 'score': score},
            ...
        ]
        """
        return [
            {'bbox': tuple(box), 'score': score}
            for box, score in zip(boxes, scores)
        ]

    def track(self, boxes, scores, frame_num):
        detections = self.gen_detections(boxes, scores)

        # apply low threshold to detections
        dets = [det for det in detections if det['score'] >= self.sigma_l]

        updated_tracks = []
        for track in self.tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = \
                    max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))

                if iou(track['bboxes'][-1], best_match['bbox']) >= self.sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = \
                        max(track['max_score'], best_match['score'])

                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= self.sigma_h \
                and len(track['bboxes']) >= self.t_min:
                    self.tracks_finished.append(track)

        # create new tracks
        new_tracks = [
            {
                'bboxes': [det['bbox']],
                'max_score': det['score'],
                'start_frame': frame_num
            }
            for det in dets
        ]
        self.tracks_active = updated_tracks + new_tracks

        return self.tracks_active

    def finish_track(self):
        """finish all remaining active tracks"""
        self.tracks_finished += [
            track for track in self.tracks_active
            if track['max_score'] >= self.sigma_h and len(track['bboxes']) >= self.t_min]

        return self.tracks_finished
