"""
"""
import os
import sys
import fire
import cv2 as cv


def run(in_file, is_video=False, save_to='ROIs'):
    """
    """
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
        assert ret, "Can't receive frame (stream end?). Exiting ..."
    else:
        im = cv.imread(in_file)

    rois = cv.selectROIs('Select ROIs', im)

    print(rois)

    with open(save_to, 'w+') as f:
        for roi in rois:
            f.write(' '.join(map(str, roi)))
            f.write('\n')

            x, y, w, h = roi
            cv.imshow(str(roi), im[y:y+h, x:x+w])

        cv.waitKey(0)
        cv.destroyAllWindows() 


if __name__ == "__main__":
    fire.Fire(run)

