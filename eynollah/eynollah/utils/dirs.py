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

    @property
    def model_dir_of_enhancement(self) -> str:
        return self.dir_models + "/eynollah-enhancement_20210425"

    @property
    def model_dir_of_binarization(self) -> str:
        return self.dir_models + "/eynollah-binarization_20210425"

    @property
    def model_dir_of_col_classifier(self) -> str:
        return self.dir_models + "/eynollah-column-classifier_20210425"

    @property
    def model_region_dir_p2(self) -> str:
        return self.dir_models + "/eynollah-main-regions-aug-rotation_20210425"

    @property
    def model_region_dir_fully_np(self) -> str:
        return self.dir_models + "/eynollah-full-regions-1column_20210425"

    @property
    def model_region_dir_fully(self) -> str:
        return self.dir_models + "/eynollah-full-regions-3+column_20210425"

    @property
    def model_page_dir(self) -> str:
        return self.dir_models + "/eynollah-page-extraction_20210425"

    @property
    def model_region_dir_p_ens(self) -> str:
        return self.dir_models + "/eynollah-main-regions-ensembled_20210425"

    @property
    def model_region_dir_p_ens_light(self) -> str:
        return self.dir_models + "/eynollah-main-regions_20220314"

    @property
    def model_textline_dir(self) -> str:
        return self.dir_models + "/eynollah-textline_20210425"

    @property
    def model_textline_dir_light(self) -> str:
        return self.dir_models + "/eynollah-textline_light_20210425"

    # FIXME: should have 'dir' in the name as well
    @property
    def model_tables(self) -> str:
        return self.dir_models + "/eynollah-tables_20210319"

    # FIXME: unused
    @property
    def model_region_dir_p(self):
        return self.dir_models + "/eynollah-main-regions-aug-scaling_20210425"

        
