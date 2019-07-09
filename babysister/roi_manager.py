""""""
import os
import csv
import cv2 as cv

from babysister.prompter import query_yes_no


class RoiManager():
    """"""
    fields = {
        'id' : int, 
        'x' : int, 
        'y' : int, 
        'w' : int, 
        'h' : int
    }

    @classmethod
    def _map_type(cls, roi):
        for fieldname, fieldtype in cls.fields.items():
            roi[fieldname] = fieldtype(roi[fieldname])
        return roi

    @classmethod
    def create_roi(cls, values):
        """"""
        roi = {}
        for fieldname, value in zip(cls.fields.keys(), values):
            roi[fieldname] = value
        return cls._map_type(roi)

    @classmethod
    def select_rois_over_image(cls, im, save_to, delimiter, quotechar):
        """"""
        rois = cv.selectROIs('Select ROIs', im)
        if len(rois) == 0:
            h, w, __ = im.shape
            rois = [[0, 0, w, h]]

        with open(save_to, 'w+', newline='') as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=cls.fields.keys(), 
                delimiter=delimiter, quotechar=quotechar, 
                quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()

            for roi_n, roi in enumerate(rois):
                x, y, w, h = roi
                values = [roi_n, x, y, w, h, -1]
                row = cls.create_roi(values)

                writer.writerow(row)
                print(row)
                #cv.imshow(str(roi), im[y:y+h, x:x+w])
            #cv.waitKey(0)
            #cv.destroyAllWindows() 

    @classmethod
    def select_rois(
        cls, in_file, is_video=False, save_to='rois.csv',
        delimiter=',', quotechar='"'
    ):
        """"""
        if os.path.isfile(save_to):
            do_ow = query_yes_no(
                'File exists: {}.\n Overwrite?'.format(save_to), default="no")
            if do_ow: 
                pass
            else:
                print('Ok, thanks. Bye')
                exit(1)

        if is_video:
            cap = cv.VideoCapture(in_file)
            assert cap.isOpened()
            ret, im = cap.read()
            assert ret, "Can't receive frame"
        else:
            im = cv.imread(in_file)

        cls.select_rois_over_image(im, save_to, delimiter, quotechar)

    @classmethod
    def read_rois(cls, rois_file, delimiter=',', quotechar='"'):
        """"""
        with open(rois_file, newline='') as csvfile:
            reader = csv.DictReader(
                csvfile, 
                fieldnames=cls.fields.keys(), 
                delimiter=delimiter, quotechar=quotechar,
                quoting=csv.QUOTE_NONNUMERIC)

            rois = list(reader)[1:]  # don't include field names
            return list(map(cls._map_type, rois))

