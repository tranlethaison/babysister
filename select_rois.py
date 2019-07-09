""""""
import fire

from babysister.roi_manager import RoiManager


if __name__ == "__main__":
    fire.Fire(RoiManager.select_rois)

