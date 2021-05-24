from tcn import TCN
from cnn_blstm import CNN_BLSTM
import torch
import torch.nn.functional as F
import preprocessor
import os
import numpy as np
import random
import scorer
import argparse

#TODO: implementar um "classificador online" para rodar em tempo real

def train(model, optimizer, x_val, y_val):
    model.train()
    x = torch.autograd.Variable(x_val, requires_grad=False).cuda()
    y = torch.autograd.Variable(y_val, requires_grad=False).cuda()
    optimizer.zero_grad()
    output = model(x)
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, x_val, y_val):
    model.eval()
    with torch.no_grad():
        x = torch.autograd.Variable(x_val, requires_grad=False).cuda()
        y = torch.autograd.Variable(y_val, requires_grad=False).cuda()
        output = model(x)
        loss = F.nll_loss(output, y, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        preds = pred.view(pred.numel())
        return preds, y, loss


def f1_score(preds, labels, class_id):
    '''
    preds: precictions made by the network
    labels: list of expected targets
    class_id: corresponding id of the class
    '''
    true_count = torch.eq(labels, class_id).sum()
    true_positive = torch.logical_and(torch.eq(labels, preds),
                                      torch.eq(labels, class_id)).sum().float()
    precision = torch.div(true_positive, torch.eq(preds, class_id).sum().float())
    precision = torch.where(torch.isnan(precision),
                            torch.zeros_like(precision).type_as(true_positive),
                            precision)
    recall = torch.div(true_positive, true_count)
    f1 = 2*precision*recall / (precision+recall)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive),f1)
    return f1.item()


def save_test_output(model_path, preds, labels):
    output_path = 'outputs/' + model_path
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    preds = preds.cpu()
    labels = labels.cpu()
    np.savez(output_path, pred=preds.numpy(), gt=labels.numpy())


def predict(model, num_test_batches, batch_size, trX_val, trY_val, pproc):
    total_pred = torch.Tensor([]).cuda()
    total_label = torch.Tensor([]).cuda()
    test_loss = 0
    test_size = len(trY_val)
    for k in range(num_test_batches):
        start, end = k*batch_size, (k+1)*batch_size
        X,Y = pproc.create_batches(trX_val, trY_val, start, end)
        preds, labels, loss = test(model, X, Y)
        test_loss += loss
        total_pred = torch.cat([total_pred, preds], dim=0)
        total_label = torch.cat([total_label, labels], dim=0)
    test_loss /= test_size
    return test_loss, total_pred, total_label


def print_scores(total_pred, total_label, test_loss):
    f1_fix = f1_score(total_pred, total_label, 0)*100
    f1_sacc = f1_score(total_pred, total_label, 1)*100
    f1_sp = f1_score(total_pred, total_label, 2)*100
    f1_blink = f1_score(total_pred, total_label, 3)*100
    print('\nTest set: Average loss: {:.4f}, F1_FIX: {:.2f}%, F1_SACC: {:.2f}%, F1_SP: {:.2f}%, F1_BLK: {:.2f}%\n'.format(
        test_loss, f1_fix, f1_sacc, f1_sp, f1_blink
    ))


def set_randomness(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_model(args, layers, features):
    if args.model == 'tcn':
        model = TCN(args.timesteps, 4, layers,
                    kernel_size=args.kernel_size, dropout=args.dropout)
        model.cuda()
        return model
    if args.model == 'cnn_blstm':
        model = CNN_BLSTM(args.timesteps, 4, args.kernel_size, args.dropout,
                          features, blstm_layers=2)
        model.cuda()
        return model


def main(args, folds=10):
    set_randomness(0)
    print("Loading data...")
    dataset = args.dataset
    proc_style = '_' + args.preprocessing
    pproc = preprocessor.Preprocessor(window_length=1, offset=0, stride=8)
    if not os.path.exists("cached/" + pproc.append_options(dataset + proc_style)):
        if dataset == 'hmr':
            pproc.process_folder_parallel('data_hmr', 'cached/hmr' + proc_style, workers=12)
        elif dataset == 'ibdt':
            pproc.process_folder_parallel('etra2016-ibdt-dataset/transformed', 'cached/ibdt', workers=12)
        elif dataset == 'gazecom':
            pproc.process_folder_parallel('data_gazecom', 'cached/gazecom' + proc_style, workers=12)
   
    #fold = pproc.load_data_k_fold_parallel('cached/'+pproc.append_options(dataset), workers=8)
    fold = pproc.load_data_k_fold('cached/'+pproc.append_options(dataset + proc_style), folds=folds)
    for fold_i in range(folds):
        trX, trY, teX, teY = next(fold)
        #breaking training data into train/dev sets
        trX, trX_val = trX[:int(len(trX)*0.9)], trX[int(len(trX)*0.9):]
        trY, trY_val = trY[:int(len(trY)*0.9)], trY[int(len(trY)*0.9):] 

        train_size = len(trY)
        test_size  = len(trY_val)
        n_classes  = 4
        features   = trX.shape[1]
        batch_size = args.batch_size
        epochs     = args.epochs
        channel_sizes = [30]*4
        timesteps  = args.timesteps
        steps = 0
        lr = 0.01
        
        model = get_model(args, channel_sizes, features)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        num_test_batches = test_size//batch_size

        for epoch in range(1, epochs+1):
            cost = 0
            num_batches = train_size//batch_size
            for k in range(num_batches):
                start, end = k * batch_size, (k+1) * batch_size
                trainX, trainY = pproc.create_batches(trX, trY, start, end) 
                cost += train(model, optimizer, trainX, trainY)
                steps += 1
                if k > 0 and k % (num_batches//10) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}\tSteps: {}'.format(
                        epoch, start, train_size,
                        100 * k / num_batches, cost/batch_size, steps 
                    ), end='\r')
                    cost = 0
            t_loss, preds, labels = predict(model, num_test_batches, batch_size, trX_val, trY_val, pproc)
            print_scores(preds, labels, t_loss)
            if epoch % 6 == 0:
                lr /= 5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        
        print(f'\nFINAL TEST - fold {fold_i+1}:\n--------------')
        t_loss, preds, labels = predict(model, num_test_batches, batch_size, teX, teY, pproc)
        print_scores(preds, labels, t_loss)
        model_param = "tcn_model_{}_BATCH-{}_LAYERS-{}_EPOCHS-{}_FOLD-{}".format(
            dataset, batch_size, len(channel_sizes), epochs, fold_i+1
        )
        save_test_output(model_param, preds, labels)
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), 'models/' + model_param + '.pt')
    model_name = model_param[:-1] if folds < 9 else model_param[:-2]
    scorer_final = scorer.Scorer('outputs/', model_name, folds=folds)
    scorer_final.score()



if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d',
                           '--dataset',
                           required=True,
                           choices=['hmr', 'gazecom'])
    argparser.add_argument('-m',
                           '--model',
                            required=True,
                            choices=['tcn', 'cnn_blstm'])
    argparser.add_argument('-b',
                           '--batch_size',
                           required=False,
                           default=2048)
    argparser.add_argument('--dropout',
                            required=False,
                            default=0.25)
    argparser.add_argument('--epochs',
                            required=False,
                            default=25)
    argparser.add_argument('--kernel_size',
                            required=False,
                            default=5)
    argparser.add_argument('-p',
                           '--preprocessing',
                           required=False,
                           default='old')
    argparser.add_argument('-t',
                           '--timesteps',
                           required=True,
                           type=int)
    args = argparser.parse_args()
    main(args)
