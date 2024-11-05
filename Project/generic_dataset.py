'''
Contaions functionallity for handeling datasets.
'''
import random
from ucimlrepo import fetch_ucirepo
import numpy as np

class GenericDataset():
    '''
    Creates a generic Dataset for running models.

    Args:
        id (int): The UCI id for the wanted dataset.
        show_info (bool): Prints info about dataset if True.

    Raises:
      ValueError: If the dataset is not meant for classification.

    '''
    def __init__(self, uci_id, train_split=1, val_split=0, seed = None, show_info = None):

        # Sets seed
        random.seed(seed)

        # Calculate test_split
        test_split = 1-train_split-val_split

        # Raise error if splits wrong
        if test_split < 0 or test_split > 1:
            raise ValueError(f"Test split should be between 0 and 1, currently {test_split}.")

        if train_split < 0 or train_split > 1:
            raise ValueError(f"Train split should be between 0 and 1, currently {train_split}.")

        if val_split < 0 or val_split > 1:
            raise ValueError(f"Validation split should be between 0 and 1, currently {val_split}.")

        # save splits.
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        # fetch dataset
        self.dataset = fetch_ucirepo(id=uci_id)

        print(type(self.dataset.data.features))
        print(type(self.dataset.data.targets))

        tup = self._split_pandas(self.dataset.data.features,train_split,val_split,test_split)
        print("\n"*5)
        print(tup[0])
        print("\n"*5)
        print(tup[1])
        print("\n"*5)
        print(tup[2])
        print("\n"*5)

        # check if Classification is one of the tasks.
        if 'Classification' not in self.dataset.metadata['tasks']:
            raise ValueError((f'This dataset is not suitable for Classification. '
                              f'Only suitable for {str(self.dataset.metadata['tasks'])}.'))

        # sets flag indicating if the database has targets.
        self.has_targets = self.dataset.data.targets is not None

        # Print info
        if show_info is True:
            print(self.__str__())


    def __str__(self):
        '''
        Returns a String with information about the dataset.

        Returns:
            info (str): Information about the dataset.
        '''
        metadata = self.dataset.metadata
        info = f"--- {str(metadata["name"])} Dataset Info ---\n"

        for key, value in metadata.items(): # Loop through metadata and append to info string.
            if key in ["intro_paper","additional_info"]:
                continue
            info += f"{str(key)}: {str(value)}\n"
        info += "-"*(len(str(metadata["name"]))+21)
        return info

    def get_metadata(self) -> dict:
        """
        Returns metadata with information about dataset.
        
        Returns:
            metadata (dict): A dict with metadata.
        """
        return self.dataset.metadata

    def get_features(self):
        """
        Returns a numpy of the dataset features
        """
        feat = self.dataset.data.features
        print(feat)
        print(feat.dtypes)
        return np.array(feat)

    def get_labels(self):
        """
        Returns a targets of the dataset features
        """
        return self.dataset.data.targets

    def _split_pandas(self, data, *splits):
        if sum(splits)!=1:
            raise ValueError("Splits should add to 1.")

        num_rows = len(data.index)

        indexes = list(range(num_rows))
        split_indexes = []

        for split in splits[:-1]:
            inxs = []
            for _ in range(int(split*num_rows)):
                inxs.append(indexes.pop(int(random.random()*len(indexes))))
            split_indexes.append(inxs)
        split_indexes.append(indexes)

        data_splits = []
        for indexes in split_indexes:
            data_splits.append(data.iloc[indexes])

        return tuple(data_splits)

if __name__ == "__main__":
    ds = GenericDataset(53, train_split=0.5, val_split=0.3, show_info=True, seed=1234)
