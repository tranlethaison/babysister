""""""
import os
import csv
import cv2 as cv

from babysister.prompter import query_yes_no


class ROIManager():
    """"""
    # field_name : default_value
    fields = {
        'id' : None, 
        'x' : None, 
        'y' : None, 
        'w' : None, 
        'h' : None
    }

    @classmethod
    def add_default(cls, roi):
        """"""
        for key, val in cls.fields.items():
            if key not in roi:
                roi[key] = val

    @classmethod
    def create_roi(cls, values):
        """"""
        roi = {}
        for fieldname, value in zip(cls.fields.keys(), values):
            roi[fieldname] = value
        return roi

    @classmethod
    def select_rois_over_image(cls, im, save_to, delimiter, quotechar):
        """"""
        print("------------------------------------------------------------")
        print("If there are no ROIs being selected,")
        print("Pressing ESC will create 1 ROI with the size of whole image.")
        print("------------------------------------------------------------")
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

            print("ROIs data:")
            for roi_n, roi_val in enumerate(rois):
                x, y, w, h = roi_val
                roi = cls.create_roi([roi_n, x, y, w, h])
                cls.add_default(roi)

                writer.writerow(roi)
                print(roi)

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

            return list(reader)[1:]  # don't include field names

