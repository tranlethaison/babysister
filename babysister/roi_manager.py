import os
import csv
import cv2 as cv

from .prompter import query_yes_no


class ROIManager:
    """ROI managing routines.

    Attributes:
        default (dict): ROI default data in {column: default} format.  
    """

    default = {"id": None, "x": None, "y": None, "w": None, "h": None}

    @classmethod
    def add_default(cls, roi_data):
        """Add default for non-existing column of `roi_data`.

        Args:
            roi_data (dict): ROI data in {column: value} format.
        """
        for key, val in cls.default.items():
            if key not in roi_data:
                roi_data[key] = val

    @classmethod
    def create_roi_data(cls, values):
        """Return a `roi_data` with `values` in {column: value} format.

        Args:
            values (list): values in the same format as `default`.
        """
        roi_data = {}
        for fieldname, value in zip(cls.default.keys(), values):
            roi_data[fieldname] = value
        return roi_data

    @classmethod
    def select_rois_over_image(cls, im, save_to, delimiter, quotechar, quoting):
        """Select ROIs over a ndarray image.

        Args:
            im (ndarray): image to select ROIs over.
            save_to (str): csv file path to save ROIs data to.
            delimiter (str): delimiter.
            quotechar (str): quote char.
            quoting (csv.QUOTE_xxx constant): quoting instruction.
        """
        print("------------------------------------------------------------")
        print("If there are no ROIs being selected,")
        print("Pressing ESC will create 1 ROI with the size of whole image.")
        print("------------------------------------------------------------")
        rois = cv.selectROIs("Select ROIs", im)
        if len(rois) == 0:
            h, w, __ = im.shape
            rois = [[0, 0, w, h]]

        with open(save_to, "w+", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=cls.default.keys(),
                delimiter=delimiter,
                quotechar=quotechar,
                quoting=quoting,
            )
            writer.writeheader()

            print("ROIs data:")
            for roi_n, roi_val in enumerate(rois):
                x, y, w, h = roi_val
                roi_data = cls.create_roi_data([roi_n, x, y, w, h])
                cls.add_default(roi_data)

                writer.writerow(roi_data)
                print(roi_data)

    @classmethod
    def select_rois(
        cls,
        in_file,
        is_video=False,
        save_to="rois.csv",
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC,
    ):
        """Select ROIs over an image or video file.

        Args:
            in_file (str): input file path.
            is_video (bool): whether `in_file` is a video file.
            save_to (str): csv file path to save ROIs data to.
            delimiter (str): delimiter.
            quotechar (str): quote char.
            quoting (csv.QUOTE_xxx constant): quoting instruction.
        """
        if os.path.isfile(save_to):
            do_ow = query_yes_no(
                "File exists: {}.\n Overwrite?".format(save_to), default="no"
            )
            if do_ow:
                pass
            else:
                print("Ok, thanks. Bye")
                exit(1)

        if is_video:
            cap = cv.VideoCapture(in_file)
            assert cap.isOpened()
            ret, im = cap.read()
            assert ret, "Can't receive frame"
        else:
            im = cv.imread(in_file)

        cls.select_rois_over_image(im, save_to, delimiter, quotechar, quoting)

    @classmethod
    def read_rois(
        cls, rois_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    ):
        """Returns ROIs data read from `rois_file`.

        Args:
            rois_file (str): ROIs data file path.
            delimiter (str): delimiter.
            quotechar (str): quote char.
            quoting (csv.QUOTE_xxx constant): quoting instruction.

        Returns:
           list of dict: ROIs data, with ROI format {column: value}.
        """
        with open(rois_file, newline="") as csvfile:
            reader = csv.DictReader(
                csvfile,
                fieldnames=cls.default.keys(),
                delimiter=delimiter,
                quotechar=quotechar,
                quoting=quoting,
            )

            return list(reader)[1:]  # don't include field names
