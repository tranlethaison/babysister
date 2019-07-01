import os
import sys
import csv
import fire
import cv2 as cv


fieldnames = ['id', 'x', 'y', 'w', 'h', 'max_objects']
fieldtypes = [int, int, int, int, int, int]


def select_rois(
    in_file, is_video=False, save_to='rois.csv',
    delimiter=',', quotechar="'"
):
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
        ret, img = cap.read()
        assert ret, "Can't receive frame"
    else:
        img = cv.imread(in_file)

    select_rois_over_image(img, save_to, delimiter, quotechar)


def select_rois_over_image(img, save_to, delimiter, quotechar):
    rois = cv.selectROIs('Select ROIs', img)

    with open(save_to, 'w+', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, 
            delimiter=delimiter, quotechar=quotechar, 
            quoting=csv.QUOTE_NONNUMERIC)

        writer.writeheader()
        for roi_n, roi in enumerate(rois):
            x, y, w, h = roi
            values = [roi_n, x, y, w, h, -1]
            row = {}
            for fieldname, value in zip(fieldnames, values):
                row[fieldname] = value
            writer.writerow(row)

            cv.imshow(str(roi), img[y:y+h, x:x+w])

        cv.waitKey(0)
        cv.destroyAllWindows() 


def read_rois(rois_file='rois.csv', delimiter=',', quotechar="'"):
    with open(rois_file, newline='') as csvfile:
        reader = csv.DictReader(
            csvfile, fieldnames=fieldnames, 
            delimiter=delimiter, quotechar=quotechar,
            quoting=csv.QUOTE_NONNUMERIC)
        
        return map_type(list(reader)[1:])  # don't return field names


def map_type(fields):
    for i in range(len(fields)):
        for fieldname, fieldtype in zip(fieldnames, fieldtypes):
            fields[i][fieldname] = fieldtype(fields[i][fieldname])
    return fields


if __name__ == "__main__":
    fire.Fire()

