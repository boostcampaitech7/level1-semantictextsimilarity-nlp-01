import pandas as pd

class AugFunction():
    def __init__(self):
        pass
    
    def empty_item(self, item):
        return pd.DataFrame(columns=item.index)
    
    def merge_items(self, items):
        return pd.DataFrame(items, columns=items[0].index)
    
    def __call__(self, item):
        raise NotImplementedError("You should implement this function")