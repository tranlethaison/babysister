""""""
import fire

from babysister.roi_manager import ROIManager


if __name__ == "__main__":
    fire.Fire(ROIManager.select_rois)

