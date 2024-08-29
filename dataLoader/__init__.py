from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .n3dv_static import N3DVDataset
from .n3dv_dynamic import N3DVDynamicDataset
from .deepview_static import DeepviewDataset
from .deepview_dynamic import DeepviewDynamicDataset


dataset_dict = {'blender': BlenderDataset,
                'llff':LLFFDataset,
                'deepview':DeepviewDataset,
                'deepview_dynamic':DeepviewDynamicDataset,
                'n3dv':N3DVDataset,
                'n3dv_dynamic':N3DVDynamicDataset,
                'tankstemple':TanksTempleDataset,
                'nsvf':NSVF,
                'own_data':YourOwnDataset}