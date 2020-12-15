from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import argparse
import random
import dataloader
from dataloader import Dataset
import numpy as np
import torch
import time
from src.model import BertForFIT
from src.optim import BertAdam
from src.utils import PYTORCH_PRETRAINED_BERT_CACHE
import functools
import yaml


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
            
def main(path_yaml):
    parser = argparse.ArgumentParser(description='BERT for FIT task')
    args = parser.parse_args()
    with open(path_yaml, 'r') as config_rd:
        config = yaml.load(config_rd)

    args.data_dir = config['data_dir']
    args.bert_model = config['bert_model']
    args.task_name = config['task_name']
    args.output_dir = config['output_dir']


    args.do_train = config['do_train']
    args.do_eval = config['do_eval']

    args.train_batch_size = config['train_batch_size']
    args.cache_size = config['cache_size']
    args.eval_batch_size = config['eval_batch_size']
    args.num_train_epochs = config['num_train_epochs']

    
    args.learning_rate = config['learning_rate']
    args.num_log_steps = config['num_log_steps']
    args.warmup_proportion = config['warmup_proportion']
    args.no_cuda = config['no_cuda']
    args.local_rank = config['local_rank']
    args.seed = config['seed']
    args.gradient_accumulation_steps = config['gradient_accumulation_steps']

    args.optimize_on_cpu = config['optimize_on_cpu']
    args.fp16 = config['fp16']
    args.loss_scale = config['loss_scale']

    args.save_model_after_epoch = config['save_model_after_epoch']

    
    assert args.do_train or args.do_eval, "ERROR"
        
    suffix = time.strftime('%Y%m%d-%H%M%S')
    args.output_dir = os.path.join(args.output_dir, suffix)
    os.makedirs(args.output_dir, exist_ok=True)
    logging = get_logger(os.path.join(args.output_dir, 'log.txt'))
    
    data_file = {
        'train' : 'train',  
        'valid' : 'valid'
        }
    for key in data_file.keys():
        data_file[key] = data_file[key] + '-' + args.bert_model + '.pt'
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            args.fp16 = False 
    assert args.gradient_accumulation_steps >= 1, "must > 1"

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    num_train_steps = None
    train_data = None
    if args.do_train:
        train_data = dataloader.Dataloader(args.data_dir, data_file['train'], args.cache_size, args.train_batch_size, device)
        num_train_steps = int(
           train_data.data_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    model = BertForFIT.from_pretrained("./pretrained/"+args.bert_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    
    global_step = 0
    if args.do_train:
        logging("[INFO] Running training")
        logging("  Batch size = {}".format(args.train_batch_size))
        logging("  Num steps = {}".format(num_train_steps))

        model.train()
        for epoch in range(int(args.num_train_epochs)):
            tr_loss = 0
            tr_acc = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for inp, tgt in train_data.__getitem__():
                loss, acc = model(inp, tgt)
                if n_gpu > 1:
                    loss = loss.mean()
                    acc = acc.sum()
                if args.fp16 and args.loss_scale != 1.0:
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                tr_acc += acc.item()
                nb_tr_examples += inp[-1].sum()
                nb_tr_steps += 1
                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1
                if (global_step % args.num_log_steps == 0):
                    logging('[INFO] Step: {} || Loss" {} || Accuracy: {}'.format(
                        global_step, tr_loss/nb_tr_examples, tr_acc/nb_tr_examples))
                    tr_loss = 0
                    tr_acc = 0
                    nb_tr_examples = 0


            if  not epoch % args.save_model_after_epoch:
                name_model = args.bert_model + '_' + str(epoch) + ".pt"
                torch.save(model.state_dict(), "./" + name_model)
    
    torch.save(model.state_dict(), f"./{args.bert_model}_final.pt")


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logging("[INFO] Running evaluation ")
        logging("  Batch size = {}".format(args.eval_batch_size))
        valid_data = dataloader.Dataloader(args.data_dir, data_file['valid'], args.cache_size, args.eval_batch_size, device)
        model.eval()
        eval_loss, eval_accuracy = 0, 0 
        nb_eval_steps, nb_eval_examples = 0, 0
        for inp, tgt in valid_data.__getitem__(shuffle=False):

            with torch.no_grad():
                tmp_eval_loss, tmp_eval_accuracy = model(inp, tgt)
            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()
                tmp_eval_accuracy = tmp_eval_accuracy.sum()
            eval_loss += tmp_eval_loss.item()
            eval_accuracy += tmp_eval_accuracy.item()
            nb_eval_examples += inp[-1].sum().item()
            nb_eval_steps += 1         

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        result = {'valid_eval_loss': eval_loss,
                  'valid_eval_accuracy': eval_accuracy,
                  'global_step': global_step}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logging("[INFO] Valid Eval results ")
            for key in sorted(result.keys()):
                logging("  {} = {}".format(key, str(result[key])))
                writer.write("%s = %s\n" % (key, str(result[key])))
                
if __name__ == "__main__":
    path_yaml = './config.yaml'
    print(time.asctime())
    main(path_yaml)
    print(time.asctime())