from cnn_lstm import CNN_LSTM
from tcn import TCN
import torch
import numpy as np
import preprocessor
import time
import argparse
import os
from sklearn.metrics import roc_curve, auc
from plotly import graph_objects as go


class Plotter():

    def __init__(self, args):
        self.args = args
        self.freq = 200 if args.dataset == 'hmr' else 250
        self.pr = preprocessor.Preprocessor(window_length=1,offset=0,
                             stride=args.strides,frequency=self.freq)
        torch.set_printoptions(sci_mode=False)
        self.n_classes = 4
        self._check_folder_exists(args.dataset)


    def _load_model(self, model_name, batch_size, features, fold):
        filename = f"{model_name}_model_{self.args.dataset}_BATCH-"
        filename += f"{batch_size}_EPOCHS-{self.args.epochs}_FOLD-"
        filename += f"{fold}.pt"
        path = os.path.join(self.args.mod, filename)
        if model_name == 'tcn':
            model = TCN(self.args.timesteps, 4, [30]*4,
                kernel_size=self.args.kernel_size, dropout=self.args.dropout)
        elif model_name == 'cnn_lstm':
            model = CNN_LSTM(self.args.timesteps, 4, self.args.kernel_size, 
                 self.args.dropout, features, self.args.lstm_layers)
        else:
            model = CNN_LSTM(self.args.timesteps, 4, self.args.kernel_size,
                 self.args.dropout, features, self.args.lstm_layers,
                 bidirectional=True)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(path))
            model.cuda()
        else:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model


    def _check_folder_exists(self, dataset):
        if not os.path.exists("cached/" + self.pr.append_options(dataset)):
            if dataset == 'hmr':
                self.pr.process_folder_parallel('data_hmr','cached/hmr', workers=12)
            elif dataset == 'gazecom':
                self.pr.process_folder_parallel('data_gazecom','cached/gazecom', workers=12)


    def predict(self, model, X_val, Y_val):
        total_pred = np.empty((0, self.n_classes), dtype=float)
        total_label = np.empty((0,), dtype=float)
        b_size = 20000
        num_test_batches = len(Y_val)//b_size
        for k in range(num_test_batches):
            start, end = k*b_size, (k+1)*b_size
            if start <= 0:
                start = self.args.timesteps
            X,Y = self.pr.create_batches(X_val, Y_val, start, end, self.args.timesteps)
            with torch.no_grad():
                output = model(X)
                layer = torch.nn.Softmax(dim=1)
                pred = layer(output)
            total_pred = np.append(total_pred, pred, axis=0)
            total_label = np.append(total_label, Y, axis=0)
        return total_pred, total_label


    def calculate_roc_auc(self, model_name, batch_size):
        print('>>> Calculating ROC AUC values for model', model_name)
        fold = self.pr.load_data_k_fold('cached/'+self.pr.append_options(self.args.dataset), 
                                        folds=self.args.folds)
        pred  = np.empty((0, self.n_classes), dtype=float) 
        label = np.empty((0,), dtype=float)
        for fold_i in range(self.args.folds):
            _, _, teX, teY = next(fold)
            features = teX.shape[1]
            model = self._load_model(model_name, batch_size, features, fold_i+1)
            preds, labels = self.predict(model, teX, teY)
            pred = np.append(pred, preds, axis=0)
            label = np.concatenate((label, labels))
        fpr, tpr = {},{}
        for c in range(self.n_classes):
            fpr_, tpr_, _ = roc_curve(label, pred[:,c], pos_label=c)
            fpr[c] = fpr_
            tpr[c] = tpr_
        all_fpr = np.unique(np.concatenate([fpr[c] for c in range(self.n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.n_classes
        roc_auc = auc(all_fpr, mean_tpr)
        return all_fpr, mean_tpr, roc_auc

    
    def plot_roc_auc(self, fpr, tpr, roc_auc):
        plots = []
        for model in fpr.keys():
            plots.append(go.Scatter(x=fpr[model], y=tpr[model], 
                        name=f"{model} AUC = {roc_auc[model]:.3f}"))
        plots.append(go.Scatter(x=[0, 1], y=[0, 1], showlegend=False,
                line=dict(dash='dash', color='gray')))
        fig = go.Figure(plots)
        fig.update_layout(
            xaxis_title='False positive rate',
            yaxis_title='True positive rate',
            title=f'ROC curves on {self.args.dataset} dataset',
            font = dict(size=16),
            legend=dict(yanchor='bottom', y=0.05, xanchor='right', x=0.98),
            title_x=0.5
        )
        fig.update_xaxes(range=[-0.02,1])
        fig.update_yaxes(range=[0,1.02])
        fig.write_image(f'fig_ROC_AUC_{self.args.dataset}.png')

                
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--models',
                        required=False,
                        default=['tcn', 'cnn_lstm', 'cnn_blstm'])
    parser.add_argument('-d',
                        '--dataset',
                        required=True)
    parser.add_argument('-b',
                        '--batch_size',
                        required=False,
                        default=[2048, 8192, 8192],
                        type=int)
    parser.add_argument('-e',
                        '--epochs',
                        required=False,
                        default=25,
                        type=int)
    parser.add_argument('-t',
                        '--timesteps',
                        required=True,
                        type=int)
    parser.add_argument('-k',
                        '--kernel_size',
                        required=False,
                        default=5,
                        type=int)
    parser.add_argument('--dropout',
                        required=False,
                        default=0.25,
                        type=float)
    parser.add_argument('--lstm_layers',
                        required=False,
                        default=2,
                        type=int)
    parser.add_argument('-f',
                        '--folds',
                        required=True,
                        default=5,
                        type=int)
    parser.add_argument('-s',
                        '--strides',
                        required=False,
                        default=8,
                        type=int)
    parser.add_argument('--out',
                        help='model outputs for all folds',
                        required=False,
                        default='final_outputs')
    parser.add_argument('--mod',
                        help='models folder',
                        required=False,
                        default='final_models')

    args = parser.parse_args()
    pltr = Plotter(args)
    tpr, fpr, roc_auc = {},{},{}
    for i, model_name in enumerate(args.models):
        fpr_, tpr_, roc_auc_ = pltr.calculate_roc_auc(model_name, args.batch_size[i])
        fpr[model_name] = fpr_
        tpr[model_name] = tpr_
        roc_auc[model_name] = roc_auc_
    pltr.plot_roc_auc(fpr, tpr, roc_auc)
    # plt.figure()
    # lw = 2
    # #color=['blue', 'red', 'orange', 'green']
    # #for i in range(4):
    # #    plt.plot(fpr[i], tpr[i], color=color[i],
    # #            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
    # plt.plot(fpr, tpr, color='orange', lw=lw,
    #     label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([-0.02, 1.0])
    # plt.ylim([0.0, 1.02])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig(f'{args.model}_{args.dataset}.png')

   


