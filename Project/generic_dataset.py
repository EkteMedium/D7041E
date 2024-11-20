'''
Contaions functionallity for handeling datasets.
'''
import random
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd

class GenericDataset():
    '''
    Creates a generic Dataset for running models.

    Args:
        id (int): The UCI id for the wanted dataset.
        show_info (bool): Prints info about dataset if True.

    Raises:
      ValueError: If the dataset is not meant for classification.

    '''
    def __init__(self, uci_id, splits = (1,0,0), seed = None, show_info = False):

        # Sets seed
        random.seed(seed)

        # Raise error if splits wrong
        if splits[0] < 0 or splits[0] > 1:
            raise ValueError(f"Train split should be between 0 and 1, currently {splits[0]}.")

        if splits[1] < 0 or splits[1] > 1:
            raise ValueError(f"Validation split should be between 0 and 1, currently {splits[1]}.")

        if splits[2] < 0 or splits[2] > 1:
            raise ValueError(f"Test split should be between 0 and 1, currently {splits[2]}.")

        if sum(splits)!=1:
            raise ValueError("Splits should add to 1.")

        # Save splits.
        self.train_split = splits[0]
        self.val_split = splits[1]
        self.test_split = splits[2]

        # Fetch dataset
        self.dataset = fetch_ucirepo(id=uci_id)

        # Sets flag indicating if the database has targets.
        self.has_targets = self.dataset.data.targets is not None

        df = self.dataset.data.features
        print(df)
        print(df.dtypes)
        df = self._pre_process(self.dataset.data.features)
        print(df)
        print(df.dtypes)

        # Split into train/val/test splits.
        self.features_tuple, self.targets_tuple = self._split_pandas(self.dataset.data.features,
                                                                     self.dataset.data.targets,
                                                                     self.train_split,
                                                                     self.val_split,
                                                                     self.test_split)

        # Check if Classification is one of the tasks.
        if 'Classification' not in self.dataset.metadata['tasks']:
            raise ValueError((f'This dataset is not suitable for Classification. '
                              f'Only suitable for {str(self.dataset.metadata['tasks'])}.'))
        

        # Print info.
        if show_info:
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

    def get_train(self):
        """
        Returns a numpy of the train features
        """
        features = self.features_tuple[0]
        targets = self.targets_tuple[0]
        return np.array(features), np.array(targets)

    def get_validation(self):
        """
        Returns a numpy of the train features
        """
        features = self.features_tuple[1]
        targets = self.targets_tuple[1]
        return np.array(features), np.array(targets)

    def get_test(self):
        """
        Returns a numpy of the train features
        """
        features = self.features_tuple[2]
        targets = self.targets_tuple[2]
        return np.array(features), np.array(targets)

    def get_labels(self):
        """
        Returns a targets of the dataset features
        """
        return self.dataset.data.targets

    def _split_pandas(self, features, targets, *splits):
        if sum(splits)!=1:
            raise ValueError("Splits should add to 1.")

        num_rows = len(features.index)

        indexes = list(range(num_rows))
        split_indexes = []

        for split in splits[:-1]:
            inxs = []
            for _ in range(int(split*num_rows)):
                inxs.append(indexes.pop(int(random.random()*len(indexes))))
            split_indexes.append(inxs)
        split_indexes.append(indexes)

        feature_splits = []
        target_splits = []
        for indexes in split_indexes:
            feature_splits.append(features.iloc[indexes])
            target_splits.append(targets.iloc[indexes])

        return (tuple(feature_splits), tuple(target_splits))

    def _pre_process(self,df):
        df = self._normalize(df)
        df = self._one_hot_encode(df)
        return df

    def _one_hot_encode(self,df:pd.DataFrame):
        object_columns = [col for col in df if df[col].dtype=='O']
        df2 = pd.get_dummies(df, columns=object_columns, dtype=float).astype(float)
        return df2

    def _normalize(self, df, means=None, stds=None):
        non_O_columns = [col for col in df if df[col].dtype!='O']

        #if non_O_columns != len(means) or non_O_columns != len(stds):
        #    raise ValueError("Length of means and stds need to be the same as number of non object columns.")

        # Whitening
        for column in non_O_columns:
            df[column] = (df[column]-df[column].mean())/df[column].std()
        
        return df

if __name__ == "__main__":
    ds = GenericDataset(53, splits=(0.5, 0.3, 0.2), show_info=True, seed=1234)
