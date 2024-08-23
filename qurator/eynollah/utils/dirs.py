from dataclasses import dataclass
from typing import Optional


@dataclass()
class EynollahDirs():
    """
    Wrapper for all the dir_ kwargs everywhere
    """
    dir_models : str
    dir_out : Optional[str] = None
    dir_in : Optional[str] = None
    dir_of_cropped_images : Optional[str] = None
    dir_of_layout : Optional[str] = None
    dir_of_deskewed : Optional[str] = None
    dir_of_all : Optional[str] = None
    dir_save_page : Optional[str] = None


