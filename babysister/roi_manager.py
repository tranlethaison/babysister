""""""
import os
import sys
import csv
import fire
import cv2 as cv


fields = {
    'id' : int, 
    'x' : int, 
    'y' : int, 
    'w' : int, 
    'h' : int, 
    'max_objects' : int
}


def _map_type(roi):
    for fieldname, fieldtype in fields.items():
        roi[fieldname] = fieldtype(roi[fieldname])
    return roi


def create_roi(values):
    """"""
    roi = {}
    for fieldname, value in zip(fields.keys(), values):
        roi[fieldname] = value
    return _map_type(roi)


def select_rois_over_image(im, save_to, delimiter, quotechar):
    """"""
    rois = cv.selectROIs('Select ROIs', im)

    with open(save_to, 'w+', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=fields.keys(), 
            delimiter=delimiter, quotechar=quotechar, 
            quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()

        for roi_n, roi in enumerate(rois):
            x, y, w, h = roi
            values = [roi_n, x, y, w, h, -1]
            row = create_roi(values)

            writer.writerow(row)
            print(row)
            #cv.imshow(str(roi), im[y:y+h, x:x+w])
        #cv.waitKey(0)
        #cv.destroyAllWindows() 


def select_rois(
    in_file, is_video=False, save_to='rois.csv',
    delimiter=',', quotechar="'"
):
    """"""
    if os.path.isfile(save_to):
        do_ow = input('File exists: {}.\n Overwrite? y/N\n'.format(save_to))
        if do_ow.lower() == 'y': 
            pass
        else:
            print('Ok, thanks. Bye')
            sys.exit(1)

    if is_video:
        cap = cv.VideoCapture(in_file)
        assert cap.isOpened()
        ret, im = cap.read()
        assert ret, "Can't receive frame"
    else:
        im = cv.imread(in_file)

    select_rois_over_image(im, save_to, delimiter, quotechar)


def read_rois(rois_file='rois.csv', delimiter=',', quotechar="'"):
    """"""
    with open(rois_file, newline='') as csvfile:
        reader = csv.DictReader(
            csvfile, fieldnames=fields.keys(), 
            delimiter=delimiter, quotechar=quotechar,
            quoting=csv.QUOTE_NONNUMERIC)

        rois = list(reader)[1:]  # don't include field names
        return list(map(_map_type, rois))


if __name__ == "__main__":
    fire.Fire()

