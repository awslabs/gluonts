from typing import Any, Dict, Iterable

# Dictionary used for data flowing through the transformations.
DataEntry = Dict[str, Any]

# A Dataset is an iterable of DataEntry.
Dataset = Iterable[DataEntry]
