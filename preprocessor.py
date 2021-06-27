import numpy as np
import pandas as pd
import torch
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


class Preprocessor():

    def __init__(self, stride=9, frequency=200, window_length=1, offset=1):
        self.offset = offset
        self.stride = stride
        self.frequency = frequency
        self.length = window_length
        self.f_len = stride*3
        self.train_X, self.test_X = np.empty((0,self.f_len)), np.empty((0,self.f_len))
        self.train_Y, self.test_Y = np.empty((0,)), np.empty((0,))


    def process_folder(self, base_path, out_path):
        '''
        Extract features from a folder containing the
        dataset. The processed files are stored in an
        optimized format for fast I/O
        '''
        out_path = self.append_options(out_path)
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
                X,Y = self.process_data(data)
                self.save_processed_file(X, Y, outfile)


    def process_folder_parallel(self, base_path, out_path, workers):
        srcs, outfiles = [], []
        out_path = self.append_options(out_path)
        for dirpath, dirnames, files in os.walk(base_path):
            for f in files:
                src = os.path.join(dirpath, f)
                srcs.append(src)
                patt = re.findall(".*\/(.*)\/[A-Z]*", src)
                outpath = os.path.join(out_path, patt[0])
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                outfile = os.path.join(out_path, patt[0], f[:-4])
                outfiles.append(outfile)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(self._process_one, srcs, outfiles, timeout=90)

    
    def _process_one(self, src, outfile):
        print(f'>>> extracting features from {src}...')
        data = self.load_file(src)
        X,Y = self.process_data(data)
        self.save_processed_file(X, Y, outfile)


    def load_processed_data(self, base_path, ratio=0.8):
        '''
        Returns X_train, Y_train, X_test, Y_test
        the split ratio is defined by the param 'ratio'
        '''
        print(">>> Loading data...")
        base_path = self.append_options(base_path)
        X_base, Y_base = None, None
        for dirpath, dirnames, files in os.walk(base_path):
            dirnames.sort()
            files.sort()
            for f in files:
                src = os.path.join(dirpath, f)
                data = np.load(src, mmap_mode='r')
                X, Y = data['X'], data['Y']
                X_base, Y_base = self._stack_data((X_base,Y_base), (X,Y))
        idx = int(ratio * X_base.shape[0])
        train_X, test_X = X_base[:idx], X_base[idx:]
        train_Y, test_Y = Y_base[:idx], Y_base[idx:]
        return train_X, train_Y, test_X, test_Y


    def _stack_data(self, base_data, new_data):
        X_base, Y_base = base_data
        X_new, Y_new = new_data
        if X_base is None and Y_base is None:
            return X_new, Y_new
        X_base = np.vstack((X_base, X_new))
        Y_base = np.concatenate((Y_base, Y_new))
        return X_base, Y_base


    def load_processed_data_parallel(self, base_path, ratio=0.8, workers=12):
        print('>>> Loading data...')
        data_array, futures = [], []
        for dirpath, dirnames, files in os.walk(base_path):
            dirnames.sort()
            files.sort()
            for f in files:
                src = os.path.join(dirpath, f)
                data = np.load(src, mmap_mode='r')
                data_array.append(data)
        n_data = self._get_chunks(data_array, workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._concatenate_chunk, data):
                       data for data in n_data}
            for future in as_completed(futures):
                train_X, train_Y, test_X, test_Y = future.result()
                self.train_X = np.vstack((self.train_X, train_X))
                self.train_Y = np.concatenate((self.train_Y, train_Y))
                self.test_X  = np.vstack((self.test_X, test_X))
                self.test_Y  = np.concatenate((self.test_Y, test_Y))
        return train_X, train_Y, test_X, test_Y


    def _get_chunks(self, data, n_chunks):
        chunks = []
        n = int(np.ceil(len(data)/n_chunks))
        for i in range(0, len(data), n):
            chunk = data[i: n+i]
            chunks.append(chunk)
        return chunks


    def _concatenate_chunk(self, n_data, ratio=0.8):
        train_X, test_X = np.empty((0,self.f_len)), np.empty((0,self.f_len))
        train_Y, test_Y = np.empty((0,)), np.empty((0,))
        for data in n_data:
            idx     = int(ratio * len(data['X']))
            train_X = np.vstack((train_X, data['X'][:idx]))
            train_Y = np.concatenate((train_Y, data['Y'][:idx]))
            test_X  = np.vstack((test_X, data['X'][idx:]))
            test_Y  = np.concatenate((test_Y, data['Y'][idx:]))       
        return train_X, train_Y, test_X, test_Y

    
    def load_data_k_fold(self, base_path, folds=5):
        '''
        Generator that returns a different split training/test
        at each call based on the number of total folds
        '''
        for k in range(folds):
            print(f'>>> Loading fold {k+1}...')
            X_base, Y_base = None, None
            for dirpath, dirnames, files in os.walk(base_path):
                dirnames.sort()
                files.sort()
                for f in files:
                    src = os.path.join(dirpath, f)
                    data = np.load(src, mmap_mode='r')
                    X, Y = data['X'], data['Y']
                    X_base, Y_base = self._stack_data((X_base,Y_base), (X,Y))
            idx_l = k*int(X_base.shape[0]/folds)
            idx_h = idx_l + int(X_base.shape[0]/folds)
            test_X = X_base[idx_l:idx_h]
            test_Y = Y_base[idx_l:idx_h]
            train_X = np.vstack((X_base[:idx_l], X_base[idx_h:]))
            train_Y = np.concatenate((Y_base[:idx_l], Y_base[idx_h:]))          
            yield train_X, train_Y, test_X, test_Y


    def load_data_k_fold_parallel(self, base_path, folds=5, workers=12):
        for k in range(folds):
            print(f'>>> Loading fold {k+1}...')
            train_X, test_X = np.empty((0,self.f_len)), np.empty((0,self.f_len))
            train_Y, test_Y = np.empty((0,)), np.empty((0,))
            data_array, futures = [], []
            for dirpath, dirnames, files in os.walk(base_path):
                dirnames.sort()
                files.sort()
                for f in files:
                    src = os.path.join(dirpath, f)
                    data = np.load(src, mmap_mode='r')
                    data_array.append(data)
            n_data = self._get_chunks(data_array, workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self._concatenate_chunk_k_fold, data):
                           data for data in n_data}
                for future in as_completed(futures):
                    tr_X, tr_Y = future.result()
                    train_X = np.vstack((train_X, tr_X))
                    train_Y = np.concatenate((train_Y, tr_Y))
            idx_l = k*int(len(train_X)/folds)
            idx_h = idx_l + int(len(train_X)/folds)
            test_X = np.vstack((test_X, train_X[idx_l:idx_h]))
            test_Y = np.concatenate((test_Y, train_Y[idx_l:idx_h]))
            train_X = np.vstack((train_X[:idx_l], train_X[idx_h:]))
            train_Y = np.concatenate((train_Y[:idx_l], train_Y[idx_h:]))
            yield train_X, train_Y, test_X, test_Y   
 

    def _concatenate_chunk_k_fold(self, n_data):
        train_X, test_X = np.empty((0,self.f_len)), np.empty((0,self.f_len))
        train_Y, test_Y = np.empty((0,)), np.empty((0,))
        for data in n_data:
            train_X = np.vstack((train_X, data['X']))
            train_Y = np.concatenate((train_Y, data['Y']))      
        return train_X, train_Y

    
    def append_options(self, outpath):
        outpath += f'_s{self.stride}'
        outpath += f'_f{self.frequency}'
        outpath += f'_w{self.length}'
        outpath += f'_o{self.offset}'
        return outpath


    def load_file(self, file_path):
        '''
        Read a single file and convert it to DataFrame
        '''
        data = pd.read_csv(file_path, sep='\t')
        if 'Filename' in data.columns:
            return data.drop(['Filename'], axis=1)
        return data


    def save_processed_file(self, X, Y, file_path):
        np.savez(file_path, X=X, Y=Y)


    def process_data(self, data):
        '''
        data: dataframe to extract features from
        stride: number of multiscale windows of powers of 2, i.e.:
                window sizes for stride 3 -> 1, 2, 4
                window sizes for stride 5 -> 1, 2, 4, 8, 16 
        target_offset: position of the target in w.r.t. the 
                       last sample of the window. E.g.,
                       an offset of 1 = next sample is the target
                       an offset of -5 = look-ahead of 5 
        '''
        strides = [2**val for val in range(self.stride)]
        fac = (self.frequency * self.length)/strides[-1]
        window = [int(np.ceil(i*fac)) for i in strides]
        latency = 1000/self.frequency #in ms
        x = data['X_coord'].to_numpy()
        y = data['Y_coord'].to_numpy()
        c = data['Confidence'].to_numpy()
        p = data['Pattern'].to_numpy()
        X,Y = self.extract_features(x,y,c,p, window, latency)
        return X, Y


    def extract_features(self, x, y, conf, targets, windows, latency):
        ini = int(np.ceil(self.frequency * self.length))
        tr_tensor  = np.zeros((len(x)-ini, 2*len(windows))) #num X sets of features
        tgt_tensor = np.zeros(len(targets)-ini,)
        for i in range(ini, len(x)):
            for j in range(len(windows)):
                start_pos, end_pos = self._get_start_end(i,windows[j])
                if start_pos == end_pos:
                    continue
                diff_x = x[end_pos] - x[start_pos]
                diff_y = y[end_pos] - y[start_pos]
                ampl   = np.math.sqrt(diff_x**2 + diff_y**2)
                time   = ((end_pos - start_pos)*latency)/1000
                #saving speed
                tr_tensor[i-ini][j] = ampl/time
                #saving direction % window
                tr_tensor[i-ini][j+len(windows)] = np.math.atan2(diff_y, diff_x)
            tgt_tensor[i-ini] = self._convert_label(targets[i+self.offset])
        return tr_tensor, tgt_tensor


    def _get_start_end(self, i, step):
        '''
        i -> the most recent position
        '''
        end_pos = i
        start_pos = i - step
        if start_pos < 0:
            start_pos = 0
        return start_pos, end_pos


    def create_batches(self, X, Y, start, end, timesteps, randomize=False):
        if timesteps == 1:
            return self._create_batches_single(X, Y, start, end, randomize)
        b_Y = Y[start-1:end-1]
        b_X = np.array([X[i-timesteps:i,:] for i in range(start, end)])
        if randomize:
            shuffler = np.random.permutation(len(b_Y))
            b_X = b_X[shuffler]
            b_Y = b_Y[shuffler]
        batch_X = torch.from_numpy(b_X).float().cuda()
        batch_Y = torch.from_numpy(b_Y).long().cuda()
        return batch_X, batch_Y



    def _create_batches_single(self, X, Y, start, end, randomize):
        b_Y = Y[start:end]
        b_X = X[start:end]
        b_X = b_X.reshape(b_X.shape[0],1,b_X.shape[1])
        batch_X = torch.from_numpy(b_X).float().cuda()
        batch_Y = torch.from_numpy(b_Y).long().cuda()
        return batch_X, batch_Y


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
    preprocessor = Preprocessor(window_length=1, offset=0, 
                                stride=8, frequency=200)
    #preprocessor.process_folder('data_gazecom','cached/gazecom')
    preprocessor.process_folder('data_hmr', 'cached/hmr')

