import numpy as np
import pandas as pd
import torch
import os
import re

class Preprocessor():

    def __init__(self, frequency=200, offset=1):
        self.window = frequency
        self.latency = 1/frequency
        self.offset = offset


    def process_folder(self, base_path, out_path):
        for dirpath, dirnames, files in os.walk(base_path):
            for f in files:
                src = os.path.join(dirpath, f)
                patt = re.findall(".*\/(.*)\/[A-Z]*", src)
                outpath = os.path.join(out_path, patt[0])
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                outfile = os.path.join(out_path, patt[0], f[:-4])
                print(f'>>> extracting features from {src}...')
                data = self.load_file(src)
                X,Y = self.process_data(data, 9, self.offset)
                self.save_processed_file(X, Y, outfile)


    def load_processed_data(self, base_path, ratio=0.8):
        '''
        Returns X_train, Y_train, X_test, Y_test
        the split ratio is defined by the param 'ratio'
        '''
        print(">>> Loading...")
        train_X, test_X = np.empty((0,28)), np.empty((0,28))
        train_Y, test_Y = np.empty((0,)), np.empty((0,))
        for dirpath, dirnames, files in os.walk(base_path):
            for f in files:
                src = os.path.join(dirpath, f)
                print(f'>>> {src}...')
                data = np.load(src, mmap_mode='r')
                idx = int(ratio * len(data['X']))
                train_X = np.vstack((train_X, data['X'][:idx]))
                train_Y = np.concatenate((train_Y, data['Y'][:idx]))
                test_X = np.vstack((test_X, data['X'][idx:]))
                test_Y = np.concatenate((test_Y, data['Y'][idx:]))
        return train_X, train_Y, test_X, test_Y

    
    def load_data_k_fold(self, base_path, folds=5):
        '''
        Generator that returns a different split training/test
        at each call based on the number of total folds
        '''
        for k in range(folds):
            print(f'>>> Loading fold {k+1}...')
            train_X, test_X = np.empty((0,28)), np.empty((0,28))
            train_Y, test_Y = np.empty((0,)), np.empty((0,))
            for dirpath, dirnames, files in os.walk(base_path):
                for f in files:
                    src = os.path.join(dirpath, f)
                    print(f'>>> {src}...')
                    data = np.load(src, mmap_mode='r')
                    train_X = np.vstack((train_X, data['X']))
                    train_Y = np.concatenate((train_Y, data['Y']))
            idx_l = k*int(len(train_X)/folds)
            idx_h = idx_l + int(len(train_X)/folds)
            test_X = np.vstack((test_X, train_X[idx_l:idx_h]))
            test_Y = np.concatenate((test_Y, train_Y[idx_l:idx_h]))
            train_X = np.vstack((train_X[:idx_l], train_X[idx_h:]))
            train_Y = np.concatenate((train_Y[:idx_l], train_Y[idx_h:]))            
            yield train_X, train_Y, test_X, test_Y



    def load_file(self, file_path):
        data = pd.read_csv(file_path, sep='\t')
        if 'Filename' in data.columns:
            return data.drop(['Filename'], axis=1)
        return data


    def save_processed_file(self, X, Y, file_path):
        np.savez(file_path, X=X, Y=Y)


    def process_data(self, data, stride, target_offset):
        '''
        data: dataframe to extract features from
        stride: number of multiscale windows of powers of 2, i.e.:
                window sizes for stride 3 -> 1, 2, 4
                window sizes for stride 5 -> 1, 2, 4, 8, 16 
        target_offsert: position of the target in w.r.t. the 
                        last sample of the window
        '''
        feat_list, target_list = [],[]
        for i in range(2**(stride-1), len(data)-1):
            features = self.extract_features(data, i, stride)
            target = self._convert_label(data.loc[i+target_offset,'Pattern'])
            feat_list.append(features)
            target_list.append(target)
        return np.array(feat_list), np.array(target_list)


    def extract_features(self, data, i, stride):
        '''
        i is the index to the rightmost position in the window
        '''
        x_ini = data.loc[i,'X_coord']
        y_ini = data.loc[i,'Y_coord']
        c_ini = data.loc[i,'Confidence']
        strides = [2**val for val in range(stride)]
        speeds, directions, confs = [],[],[c_ini]
        for j in strides:
            pos = i-j
            x_end = data.loc[pos, 'X_coord']
            y_end = data.loc[pos, 'Y_coord']
            p1, p2 = (x_ini,y_ini), (x_end,y_end)
            speed, direc = self.calculate_features(p1,p2,j)
            speeds.append(speed)
            directions.append(direc)
            confs.append(data.loc[pos, 'Confidence'])
        return np.array(speeds + directions + confs)


    def calculate_features(self, p1, p2, delta_t):
        diff_x = p1[0]-p2[0]
        diff_y = p1[1]-p2[1]
        displ = np.math.sqrt(diff_x**2 + diff_y**2)
        speed = (displ/delta_t) * 1000
        direc = np.math.atan2(diff_y, diff_x)
        return speed, direc


    def _convert_label(self, target):
        if target == 'F':
            return 0
        if target == 'S':
            return 1
        if target == 'P':
            return 2
        if target == 'B':
            return 3
        


if __name__=='__main__':
    preprocessor = Preprocessor()
    #preprocessor.process_folder('data/', 'cached/hmr/')
    preprocessor.process_folder('etra2016-ibdt-dataset/transformed/', 'cached/ibdt/')
    #preprocessor.load_processed_data('cached/')

