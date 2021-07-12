from cnn_lstm import CNN_LSTM
from tcn import TCN
import torch
import numpy as np
import preprocessor
import time
import argparse
import os
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from plotly import graph_objects as go


class Plotter():

    def __init__(self, args):
        self.args = args
        self.freq = 200 if args.dataset == 'hmr' else 250
        self.pr = preprocessor.Preprocessor(window_length=1,offset=args.offset,
                             stride=args.strides,frequency=self.freq)
        torch.set_printoptions(sci_mode=False)
        self.n_classes = 4
        self._check_folder_exists(args.dataset)


    def _get_model_path(self, model_name, batch_size, fold):
        filename = f"{model_name}_model_{self.args.dataset}_BATCH-"
        filename += f"{batch_size}_EPOCHS-{self.args.epochs}_FOLD-{fold}"
        return filename


    def _load_model(self, model_name, batch_size, features, fold):
        filename = self._get_model_path(model_name, batch_size, fold) + '.pt'
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
        b_size = 10000
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
            print('>>> Performing predictions...')
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
            autosize=False,
            width=1000,height=800,
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


    def _count_event(self, preds, gt):
        i = 0
        event_preds, event_gt = [], []
        while i < len(gt):
            g_0 = g_n = int(gt[i])
            ini, end = i, i
            while g_0 == g_n and i < len(gt):
                g_n = int(gt[i])
                end = i
                i += 1
            if ini == end:
                i += 1
                continue
            pred_event = np.array(preds[ini:end], dtype=int)
            event_preds.append(np.bincount(pred_event).argmax())
            event_gt.append(g_0)
        return event_gt, event_preds


    def score(self, model_name, batch_size, path_param):
        sample_preds, sample_gt = [],[]
        for fold in range(self.args.folds):
            path = self._get_model_path(model_name, batch_size, fold+1) + '.npz'
            path = os.path.join(self.args.out, path_param, path)
            data = np.load(path, mmap_mode='r')
            sample_preds += data['pred'].tolist()
            sample_gt += data['gt'].tolist()
        event_gt, event_preds = self._count_event(sample_preds, sample_gt)
        target_names = ['Fixations', 'Saccades', 'Pursuits', 'Noise/Blink']
        sample_report = metrics.classification_report(sample_gt, sample_preds, 
                        target_names=target_names, digits=3, output_dict=True)
        event_report = metrics.classification_report(event_gt, event_preds, 
                        target_names=target_names, digits=3, output_dict=True)
        return sample_report, event_report
    

    def plot_look_ahead(self, sample_scores, event_scores, offsets):
        color = ['blue', 'red', 'green']
        plots = []
        for i, model in enumerate(sample_scores.keys()):
            plots.append(go.Scatter(x=offsets, y=sample_scores[model],
                         name=f'{model} (sample-level)', line=dict(color=color[i])))
            plots.append(go.Scatter(x=offsets, y=event_scores[model],
                    name=f'{model} (event-level)', line=dict(color=color[i], dash='dash')))   
        fig = go.Figure(plots)
        fig.update_layout(
            autosize=False,
            width=1000,height=800,
            xaxis_title='Look-ahead (ms)',
            yaxis_title='F-score (%)',
            title=f'Look-ahead F-scores for {self.args.dataset} dataset',
            font = dict(size=16),
            xaxis = dict(tickmode='array', tickvals=offsets),
            title_x=0.5
        )
        fig.write_image(f'fig_look_ahead_{self.args.dataset}.png')

    
    def plot_timesteps(self, sample_scores, event_scores, timesteps):
        color = ['blue', 'red', 'green']
        plots = []
        for i, model in enumerate(sample_scores.keys()):
            plots.append(go.Scatter(x=timesteps, y=sample_scores[model],
                         name=f'{model} (sample level)', line=dict(color=color[i])))
            plots.append(go.Scatter(x=timesteps, y=event_scores[model],
                    name=f'{model} (event level)', line=dict(color=color[i], dash='dash')))   
        fig = go.Figure(plots)
        fig.update_layout(
            autosize=False,
            width=1000,height=800,
            xaxis_title='Timesteps (into the past)',
            yaxis_title='F-score (%)',
            title=f'F-scores with different timesteps for {self.args.dataset} dataset',
            font = dict(size=16),
            xaxis = dict(tickmode='array', tickvals=timesteps),
            title_x=0.5
        )
        fig.write_image(f'fig_timesteps_{self.args.dataset}.png')
                
            

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
    parser.add_argument('-o',
                        '--offset',
                        required=False,
                        default=0)
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

    #=========== ROC CURVES
    # tpr, fpr, roc_auc = {},{},{}
    # for i, model_name in enumerate(args.models):
    #     fpr_, tpr_, roc_auc_ = pltr.calculate_roc_auc(model_name, args.batch_size[i])
    #     fpr[model_name] = fpr_
    #     tpr[model_name] = tpr_
    #     roc_auc[model_name] = roc_auc_
    # pltr.plot_roc_auc(fpr, tpr, roc_auc)


    #============ LOOK-AHEAD
    # offsets = [0, -20,-40,-60,-80]
    # samp_scores = {m:[] for m in args.models}
    # evt_scores = {m:[] for m in args.models}
    # for offset in offsets:
    #     args.offset = offset
    #     pltr = Plotter(args)
    #     print('>>> offset:', offset)
    #     for i, model_name in enumerate(args.models):
    #         path_param = f'o_{offset}ms'
    #         sample, event = pltr.score(model_name, args.batch_size[i], path_param)
    #         samp_scores[model_name].append(sample['macro avg']['f1-score']*100)
    #         evt_scores[model_name].append(event['macro avg']['f1-score']*100)
    # pltr = Plotter(args)
    # pltr.plot_look_ahead(samp_scores, evt_scores, offsets)


    #============ TIMESTEPS
    # timesteps = ['1', '20', '50', '100', '200']
    # samp_scores = {m:[] for m in args.models}
    # evt_scores = {m:[] for m in args.models}
    # if args.dataset == 'gazecom':
    #     timesteps = ['1', '25', '62', '125', '250']
    # for t in timesteps:
    #     args.timesteps = t
    #     pltr = Plotter(args)
    #     print('>>> timesteps:', t)
    #     for i, model_name in enumerate(args.models):
    #         path_param = 't' + args.timesteps
    #         sample, event = pltr.score(model_name, args.batch_size[i], path_param)
    #         samp_scores[model_name].append(sample['macro avg']['f1-score']*100)
    #         evt_scores[model_name].append(event['macro avg']['f1-score']*100)
    # pltr = Plotter(args)
    # pltr.plot_timesteps(samp_scores, evt_scores, timesteps)



   


