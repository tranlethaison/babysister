"""
"""
import os
import sys
import fire
import cv2 as cv


def run(im_file, save_to='ROIs'):
    """
    """
    if os.path.isfile(save_to):
        do_ow = input('File exists: {}.\n Overwrite? y/N\n'.format(save_to))
        if do_ow.lower() == 'y': 
            pass
        else:
            print('Ok, thanks. Bye')
            sys.exit(1)

    im = cv.imread(im_file)
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

