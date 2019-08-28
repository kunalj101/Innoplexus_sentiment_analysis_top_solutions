import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
from pytorch_pretrained_bert import BertModel
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from sklearn.model_selection import StratifiedKFold
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from torch.utils.data.dataset import Subset
import numpy as np
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            print('*****************************************************in bert')
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.pretrained_bert_state_dict = bert.state_dict()
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            print('*****************************************************not in bert')
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
            else:
                self.model.bert.load_state_dict(self.pretrained_bert_state_dict)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
#                 print(f'head of outputs = {outputs}')
#                 print(f'head of targets = {targets}')

                loss = criterion(outputs, targets)
                         
#                 loss = self.f1_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            
            val_loss,val_acc, val_f1, val_p = self._evaluate_acc_f1(criterion,val_data_loader)
            logger.info('> val_loss: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}'.format(val_loss, val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_temp'.format(self.opt.model_name, self.opt.dataset)
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path, val_p

    def _evaluate_acc_f1(self,criterion, data_loader, test_flag=False):
        n_correct, n_total,val_loss_total = 0, 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                
                val_loss = criterion(t_outputs, t_targets)
                val_loss_total += val_loss.item() * len(t_outputs)
                
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return val_loss_total, acc, f1, t_outputs_all
    
    def _evaluate_acc_f1_loss(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return 1-f1
    

    def run(self):
        if(self.opt.give_weights):
            ws = self.opt.assign_weights
            weis= str.split(ws[0], ",")
            weights = [float(x) for x in weis] #[ 1 / number of instances for each class]
            print(weights)
            if self.opt.device.type == 'cuda':
                class_weights = torch.FloatTensor(weights).cuda()
            else:
                class_weights = torch.FloatTensor(weights)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
#         criterion = categorical_crossentropy()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        valset_len = len(self.trainset) // self.opt.cross_val_fold
        skf = StratifiedKFold(n_splits=self.opt.cross_val_fold,random_state=self.opt.seed, shuffle=False)
#         splitedsets = random_split(self.trainset, tuple([valset_len] * (self.opt.cross_val_fold - 1) + [len(self.trainset) - valset_len * (self.opt.cross_val_fold - 1)]))
        oof_preds = np.zeros((len(self.trainset), 3))
        test_pred = list()

        all_test_acc, all_test_f1 = [], []
        target = []
        for i in range(self.trainset.__len__()):
            target.append(self.trainset.__getitem__(i)['polarity'])
        for fid, (train_id,val_id) in enumerate(skf.split(target,target)):
            logger.info('fold : {}'.format(fid))
            logger.info('>' * 100)
            trainset = Subset(self.trainset, train_id)
            valset = Subset(self.trainset, val_id)
#             trainset = ConcatDataset([x for i, x in enumerate(splitedsets) if i != fid])
#             valset = splitedsets[fid]
            train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True)
            val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size, shuffle=False)

            self._reset_params()
            best_model_path,val_p = self._train(criterion, optimizer, train_data_loader, val_data_loader)
            oof_preds[val_id] = val_p.cpu().data.numpy()
            self.model.load_state_dict(torch.load(best_model_path))
            _,test_acc, test_f1, test_p = self._evaluate_acc_f1(criterion, test_data_loader)
            all_test_acc.append(test_acc)
            all_test_f1.append(test_f1)
            test_pred.append(test_p.cpu().data.numpy())
        test_pred = np.mean(test_pred, axis = 0)
        if not os.path.exists('output_final'):
            os.mkdir('output_final')
        numpy.save('output_final/{0}_{1}_{2}_{3}_{4}_{5}_test_preds.npy'.format(self.opt.model_name, self.opt.dataset,self.opt.learning_rate,self.opt.assign_weights,self.opt.num_epoch,self.opt.seed), test_pred)
        numpy.save('output_final/{0}_{1}_{2}_{3}_{4}_{5}_oof_preds.npy'.format(self.opt.model_name, self.opt.dataset,self.opt.learning_rate,self.opt.assign_weights,self.opt.num_epoch,self.opt.seed),oof_preds )

        mean_test_acc, mean_test_f1 = numpy.mean(all_test_acc), numpy.mean(all_test_f1)
        logger.info('>' * 100)
        logger.info('>>> mean_test_acc: {:.4f}, mean_test_f1: {:.4f}'.format(mean_test_acc, mean_test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='innoplex', type=str, help='innoplex, innoplex_sent')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 1e-5')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try 10')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16 or 32 BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=5, type=int, help='set seed for reproducibility')
    parser.add_argument('--cross_val_fold', default=10, type=int, help='k-fold cross validation')
    parser.add_argument('--give_weights', default=0, type=int, help='give weights 1,0')
    parser.add_argument('--assign_weights', default=None, nargs='*', help='tell weights 0.2,0.3')

    
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
    }
    dataset_files = {
        'innoplex_sent':{
            'train': '../input/train_sent.raw',
            'test': '../input/test_sent.raw'
        },
        'innoplex':{
            'train': '../input/train.raw',
            'test': '../input/test.raw'
        },
   }
    input_colses = {
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    if not os.path.exists('log'):
        os.mkdir('log')
    log_file = 'log/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()