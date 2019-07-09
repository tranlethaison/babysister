""""""
import csv
import cv2 as cv


fields = {
    'id' : int, 
    'x' : int, 
    'y' : int, 
    'w' : int, 
    'h' : int
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
    if len(rois) == 0:
        h, w, __ = im.shape
        rois = [[0, 0, w, h]]

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


def read_rois(rois_file='rois.csv', delimiter=',', quotechar='"'):
    """"""
    with open(rois_file, newline='') as csvfile:
        reader = csv.DictReader(
            csvfile, 
            fieldnames=fields.keys(), 
            delimiter=delimiter, quotechar=quotechar,
            quoting=csv.QUOTE_NONNUMERIC)

        rois = list(reader)[1:]  # don't include field names
        return list(map(_map_type, rois))

