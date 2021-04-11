import pandas as pd
import os
import re
from arff_helper import ArffHelper

#----------------------
class Converter_IBDT():

    def __init__(self):
        self.w = 640
        self.h = 480

    def process_folder(self, out_path):
        for dirpath, dirnames, files in os.walk('etra2016-ibdt-dataset/data'):
            for f in files:
                if f[-4:] == '.csv':
                    continue
                src = os.path.join(dirpath, f)
                src_label = os.path.join(dirpath, 'reviewed'+f[-9:-4]+'.csv')
                print(f'>>> converting {src} and {src_label}...')
                patt = re.findall(".*\/(.*)\/[A-Z]*", src)
                outpath = os.path.join(out_path, patt[0])
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                outfile = os.path.join(out_path, patt[0], f[:-4]+'.txt')    
                data, labels = self.load_files(src, src_label)
                to_save = self.create_file(data, labels)
                with open(outfile, 'w') as f:
                    print('saving:', outfile)
                    f.write(to_save)

    def load_files(self, file_path, label_path):
        data = pd.read_csv(file_path, sep='\t')
        labels = pd.read_csv(label_path, sep=',')
        if 'Filename' in data.columns:
            return data.drop(['Filename'], axis=1)
        return data, labels

    def create_file(self, data, labels):
        new_file = "X_coord\tY_coord\tConfidence\tPattern\n"
        for _, row in data.iterrows():
            ts = row['timestamp']
            confidence = str(row['eye_valid'])
            x = float(row['eye_x'])/640.0
            y = float(row['eye_y'])/480.0 
            label = int(labels.loc[labels['timestamp']==ts, 'label'])
            patt = self.convert_label(label)
            new_file += str(x)+'\t'+str(y)+'\t'+confidence+'\t'+patt+'\n'
        return new_file

    def convert_label(self, label):
        if label == 0:
            return 'F'
        if label == 1:
            return 'S'
        if label == 2:
            return 'P'
        if label == 3:
            return 'B'


#----------------------
class Converter_ARFF():

    def __init__(self):
        self.width  = None
        self.height = None

    def process_folder(self, base_path, out_path):
        for dirpath, dirnames, files in os.walk(base_path):
            for f in files:
                src = os.path.join(dirpath, f)
                patt = re.findall(".*\/(.*)\/[A-z]*", src)
                outpath = os.path.join(out_path, patt[0])
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                outfile = os.path.join(out_path, patt[0], f[:-3]+'.csv')  
                print(f">>> Converting {src} to CSV...")  
                output = self.convert_file(src) 
                with open(outfile, 'w') as f:
                    f.write(output)


    def convert_file(self, file_path):
        arff_obj = ArffHelper.load(open(file_path, 'r'))
        self.width  = float(arff_obj['metadata']['width_px'])
        self.height = float(arff_obj['metadata']['height_px'])
        x,y,conf,label = self.get_attr_window(arff_obj['attributes'])
        csv = "X_coord\tY_coord\tConfidence\tPattern\n"
        data = arff_obj['data']
        for i in range(len(data)):
            csv += self.convert_line(data[i], x, y, conf, label)
        return csv


    def convert_line(self, data, x, y, conf, label):
        xi = float(data[x])/self.width
        yi = float(data[y])/self.height
        ci = str(data[conf])
        pi = self.convert_label(float(data[label]))
        return str(xi)+'\t'+str(yi)+'\t'+ci+'\t'+pi+'\n'


    def convert_label(self, label):
        if label == 1:
            return 'F'
        if label == 2:
            return 'S'
        if label == 3:
            return 'P'
        return 'B'


    def get_attr_window(self, attributes):
        x, y, conf, label = 0,0,0,0
        for i in range(len(attributes)):
            if attributes[i][0] == 'x':
                x = i
            elif attributes[i][0] == 'y':
                y = i
            elif attributes[i][0] == 'confidence':
                conf = i
            elif attributes[i][0] == 'handlabeller_final':
                label = i
        return x, y, conf, label



if __name__=='__main__':
    converter = Converter_ARFF()
    converter.process_folder('gazecom_arff', 'data_gazecom')

    

