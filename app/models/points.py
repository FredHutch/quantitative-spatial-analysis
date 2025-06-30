from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from typing import Literal
import plotly.express as px

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
    clusters: Optional[pd.Series]
    xcol: str
    ycol: str
    meta_cols: List[str]
    dataset: SpatialDataset

    def plotly_scatter(self, width: int, height: int, opacity: float):
        return px.scatter(
            self.coords,
            x=self.xcol,
            y=self.ycol,
            color=(
                None if self.clusters is None else
                self.clusters.apply(str)
            ),
            width=width,
            height=height,
            opacity=opacity,
            labels={
                self.xcol: "X Coordinate",
                self.ycol: "Y Coordinate"
            }
        )


@dataclass
class SpatialRegion:
    outline: List[dict]
    dataset: SpatialDataset
    region_id: Optional[str] = None
