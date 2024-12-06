from app.cirro import list_datasets


class SpatialDataAnalyses:

    def __init__(self):
        # Get the list of datasets available from the selected project
        self.datasets = list_datasets()
