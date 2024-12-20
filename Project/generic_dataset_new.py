'''
Contaions functionallity for handeling datasets.
'''
import random
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.copy_on_write = True

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

        df_features = self.dataset.data.features
        df_label = self.dataset.data.targets

        df_features = df_features.loc[:,~df_features.columns.duplicated()].copy()
        df_label = df_label.loc[:,~df_label.columns.duplicated()].copy()

        num_non_O = len([col for col in df_features if df_features[col].dtype!='O'])

        object_columns = [col for col in df_features if df_features[col].dtype=='O']
        df_features = pd.get_dummies(df_features, columns=object_columns, dtype=float).astype(float)
        
        object_columns = [col for col in df_label if df_label[col].dtype=='O']
        for object_column in object_columns:
            df_label[object_column]=pd.DataFrame(pd.factorize(df_label[object_column].to_numpy())[0])

        df = df_features
        df['label'] = np.asarray(df_label)

        df = df[~pd.isnull(df).any(axis=1)]

        train_val, test = train_test_split(df, test_size=self.test_split)
        train, val = train_test_split(train_val,  test_size=self.val_split/(self.val_split+self.train_split))

        test_label = np.asarray(test['label'])
        test = np.asarray(test.drop(['label'],axis=1)).copy()
        np.seterr(divide='raise')
        try:
            test[:,:num_non_O] = (test[:,:num_non_O]-np.mean(test[:,:num_non_O],axis=0,keepdims=True))/np.std(test[:,:num_non_O],axis=0,keepdims=True)
        except:
            test[:,:num_non_O] = (test[:,:num_non_O]-np.mean(test[:,:num_non_O],axis=0,keepdims=True))
        self.test = (test,test_label)

        train_label = np.asarray(train['label'])
        train = np.asarray(train.drop(['label'],axis=1)).copy()
        try:
            train[:,:num_non_O] = (train[:,:num_non_O]-np.mean(train[:,:num_non_O],axis=0,keepdims=True))/np.std(train[:,:num_non_O],axis=0,keepdims=True)
        except:
            train[:,:num_non_O] = (train[:,:num_non_O]-np.mean(train[:,:num_non_O],axis=0,keepdims=True))
        self.train = (train,train_label)

        val_label = np.asarray(val['label'])
        val = np.asarray(val.drop(['label'],axis=1)).copy()
        try:
            val[:,:num_non_O] = (val[:,:num_non_O]-np.mean(val[:,:num_non_O],axis=0,keepdims=True))/np.std(val[:,:num_non_O],axis=0,keepdims=True)
        except:
            val[:,:num_non_O] = (val[:,:num_non_O]-np.mean(val[:,:num_non_O],axis=0,keepdims=True))
        self.val = (val,val_label)

        self.dataset.metadata["num_classes"] = np.max(np.asarray(df_label))+1

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

    def get_train(self, onehot=False):
        """
        Returns a numpy of the train features
        """
        return self.train

    def get_validation(self, onehot=False):
        """
        Returns a numpy of the train features
        """
        return self.val

    def get_test(self, onehot=False):
        """
        Returns a numpy of the train features
        """
        return self.test

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

        return (list(feature_splits), list(target_splits))

    def _pre_process(self,df):
        df = self._normalize(df)
        df = self._one_hot_encode(df)
        return df

    def _one_hot_encode(self,df:pd.DataFrame):
        object_columns = [col for col in df if df[col].dtype=='O']
        df2 = pd.get_dummies(df, columns=object_columns, dtype=float).astype(float)
        return df2

    def _normalize(self, df:pd.DataFrame, means=None, stds=None):
        non_O_columns = [col for col in df if df[col].dtype!='O']

        #if non_O_columns != len(means) or non_O_columns != len(stds):
        #    raise ValueError("Length of means and stds need to be the same as number of non object columns.")

        df[non_O_columns] = (df[non_O_columns] - df[non_O_columns].mean())/df[non_O_columns].std()
        
        return df

if __name__ == "__main__":
    ds = GenericDataset(53, splits=(0.5, 0.3, 0.2), show_info=True, seed=1234)
