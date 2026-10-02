"""Read only the retained historical input, checking its experiment hash."""
from pathlib import Path
import hashlib,json
import pandas as pd

def load_data(directory):
    directory=Path(directory)
    config=json.loads((directory/'configuration.json').read_text())
    path=directory.parent/'data/train.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==config['data_sha256']
    data=pd.read_csv(path)
    assert data.election.max()<=2019
    return data
