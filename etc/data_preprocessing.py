
import pandas as pd
import numpy as np
import itertools

import math
import ast
import os
import logging
import configparser as cp
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action = 'ignore', category = DataConversionWarning)

# Change str to float
def is_column_line(str):
    try:
        float(str)
        return False
    except ValueError:
        return True

    def read_csv_data(self):
        data = pd.read_csv(file_path)

    def add_invers_features(data):
        data_matrix = pd.DataFrame.as_matrix(data)
        features_names = list(data.columns.values)
        features_dict = dict(data)
        features_num = data_matrix.shape[1]
        new_features_names = []
        new_features_dict = {}
        for i in range(features_num):
            feature_name = features_names[i]
            print(feature_name)
            new_features_names.append(feature_name)
            new_features_dict[feature_name] = list(features_dict.get(feature_name))
            inverse_feature_name = 'inverse_'+feature_name
            new_features_names.append(inverse_feature_name)
            try:
                new_features_dict[inverse_feature_name] = list(1/np.array(new_features_dict[feature_name]))
            except ValueError:
                new_features_dict[inverse_feature_name] = list(np.zeros(data_matrix.shape[0]))
        return new_features_dict

    # Adds JUST the terms of the specified degree
    def add_combinatorial_terms(data, degree):
        new_features_dict = {}
        data_matrix = pd.DataFrame.as_matrix(data)
        features_names = list(data.columns.values)
        indices = list(range(features_num))
        combs = list(itertools.combinations_with_replacement(indices, degree))

        for cc in combs:
            temp = 1
            for i in range(len(cc)):
                    temp = np.multiply(temp,data_matrix[:, cc[i]])
            new_feature_value = temp.reshape((data_matrix.shape[0], 1))
            #new_feature_value=np.multiply(data_matrix[:, cc[0]],data_matrix[:, cc[1]])
            data_matrix = np.append(data_matrix , new_feature_value, axis=1)
            new_feature_name = ''
            for i in range(len(cc)-1):
                new_feature_name = new_feature_name+features_names[cc[i]]+'_'
            new_feature_name = new_feature_name+features_names[cc[i+1]]
            features_names.append(new_feature_name)
            new_features_dict[new_feature_name] = list(new_feature_value.reshape(1,data_matrix.shape[0]))

        #return features_names,data_matrix
        return new_features_dict

    # Adds ALL the terms up to the specified degree
    def add_all_comb(data,degree):
        new_features_dict = {}
        data_matrix = pd.DataFrame.as_matrix(data)
        features_names = list(data.columns.values)
        indices = list(range(features_num))

        for j in range(2, degree + 1):
            combs = list(itertools.combinations_with_replacement(indices, j))

            for cc in combs:
                temp = 1
                for i in range(len(cc)):
                        temp = np.multiply(temp,data_matrix[:, cc[i]])
                new_feature_value = temp.reshape((data_matrix.shape[0], 1))
                #new_feature_value=np.multiply(data_matrix[:, cc[0]],data_matrix[:, cc[1]])
                data_matrix = np.append(data_matrix , new_feature_value, axis=1)
                new_feature_name = ''
                for i in range(len(cc)-1):
                    new_feature_name = new_feature_name+features_names[cc[i]]+'_'
                new_feature_name = new_feature_name+features_names[cc[i+1]]
                features_names.append(new_feature_name)
                new_features_dict[new_feature_name] = list(new_feature_value.reshape(1,data_matrix.shape[0]))

            #return features_names,data_matrix
            return new_features_dict






