'''
Contaions functionallity for handeling datasets.
'''
import random
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class GenericDataset():
    '''
    Creates a generic Dataset for running models.

    Args:
        id (int): The UCI id for the wanted dataset.
        show_info (bool): Prints info about dataset if True.

    Raises:
      ValueError: If the dataset is not meant for classification.

    '''
    @np.errstate(all='ignore')
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

        # Create a copy
        df_features = df_features.loc[:,~df_features.columns.duplicated()].copy()
        df_label = df_label.loc[:,~df_label.columns.duplicated()].copy()

        # Get the number of non objective columns (eg. int or float)
        num_non_O = len([col for col in df_features if df_features[col].dtype!='O'])

        # Get all object column names and onehot encode them
        object_columns = [col for col in df_features if df_features[col].dtype=='O']
        df_features = pd.get_dummies(df_features, columns=object_columns, dtype=float).astype(float)
        
        # Go through the labels and turn categorical data to numbers
        object_columns = [col for col in df_label if df_label[col].dtype=='O']
        for object_column in object_columns:
            b = pd.factorize(df_label[object_column].to_numpy())[0]
            df_label = df_label.drop(object_column,axis=1)
            df_label = df_label.astype('float')
            df_label[object_column]=b
        
        # Shift so all labels are positive
        non_object_columns = [col for col in df_label if df_label[col].dtype!='O']
        for non_object_coloumn in non_object_columns:
            df_label[non_object_coloumn] = df_label[non_object_coloumn]-df_label[non_object_coloumn].min()

        # Combine features and labels
        df = df_features
        df['label'] = np.asarray(df_label)

        # Remove all rows with a nan
        df = df[~pd.isnull(df).any(axis=1)]
        label_arr = df['label'].to_numpy()
        df = df.drop('label',axis=1)
        df['label'] = label_arr.astype('int')
        np.errstate(all='ignore')

        # Split into train/val/test sets
        train_val, test = train_test_split(df, test_size=self.test_split)
        train, val = train_test_split(train_val,  test_size=self.val_split/(self.val_split+self.train_split))

        # Normalize all non label and non onehot train samples with train mean/std
        train_label = np.asarray(train['label'])
        train = np.asarray(train.drop(['label'],axis=1)).copy()

        mean = np.mean(train[:,:num_non_O],axis=0,keepdims=True)
        std = np.std(train[:,:num_non_O],axis=0,keepdims=True)
        train[:,:num_non_O] = np.where(std!=0,(train[:,:num_non_O]-mean)/std,(train[:,:num_non_O]-mean))
        self.train = (train,train_label)

        # Normalize all non label and non onehot val samples with train mean/std
        val_label = np.asarray(val['label'])
        val = np.asarray(val.drop(['label'],axis=1)).copy()
        val[:,:num_non_O] = np.where(std!=0,(val[:,:num_non_O]-mean)/std,(val[:,:num_non_O]-mean))
        self.val = (val,val_label)

        # Normalize all non label and non onehot test samples with train mean/std
        test_label = np.asarray(test['label'])
        test = np.asarray(test.drop(['label'],axis=1)).copy()
        test[:,:num_non_O] = np.where(std!=0,(test[:,:num_non_O]-mean)/std,(test[:,:num_non_O]-mean))
        self.test = (test,test_label)

        self.dataset.metadata["num_features"] = train.shape[1]
        self.dataset.metadata["num_classes"] = (np.max(np.asarray(df['label']))+1).astype(int).item()

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

    def get_val(self, onehot=False):
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

if __name__ == "__main__":
    ds = GenericDataset(53, splits=(0.5, 0.3, 0.2), show_info=True, seed=1234)
