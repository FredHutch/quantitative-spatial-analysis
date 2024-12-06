from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from typing import Literal

SpatialDatasetType = Literal['xenium', 'stardist']


@dataclass
class CirroDataset:
    domain: str
    project: str
    dataset: str
    path: str


@dataclass
class SpatialDataset:
    type: SpatialDatasetType
    uri: str
    cirro_source: Optional[CirroDataset] = None


@dataclass
class SpatialPoints:
    coords: pd.DataFrame
    xcol: str
    ycol: str
    meta_cols: List[str]
    dataset: SpatialDataset


@dataclass
class SpatialRegion:
    outline: List[dict]
    dataset: SpatialDataset
    region_id: Optional[str] = None
