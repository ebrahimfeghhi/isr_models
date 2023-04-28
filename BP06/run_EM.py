import numpy as np
from logging import logProcesses
import os
from statistics import mean
import torch 
from torch.utils.data import DataLoader
from datasets import OneHotLetters
from RNNcell import RNN_one_layer_EM
from run_test_trials_EM import run_test_trials_EM
torch.set_num_threads(4)
import wandb
from simulation_one import simulation_one
device = torch.device("cpu")
import argparse
from utils import encoding_policy_presentation_recall

save_model_path = '/home3/ebrahim/isr/isr_model_review/BP06/saved_models/'

parser = argparse.ArgumentParser()
parser.add_argument('--hs', type=int, default=200,
                    help="hidden_size")
parser.add_argument('--lr', type=float, default=0.001,
                    help="learning_rate")
parser.add_argument('--h0_init_val', type=float, default=0.5,
                    help="init firing rates")
parser.add_argument('--nonlin', type=str, default='sigmoid', 
                    help='Nonlinearity to use for hidden activations')
parser.add_argument('--noise_std', type=float, default=0.0, 
                    help='Standard deviation of gaussian noise')
parser.add_argument('--stopping_criteria', type=float, default=0.58, 
                    help='accuracy on lists of length six')
parser.add_argument('--clip_factor', type=float, default=1e6, 
                    help='grad_clip_vals')
parser.add_argument('--clipping', type=bool, default=False, 
                    help='whether or not to perform grad clipping')
parser.add_argument('--test_list_length', type=int, default=6, 
                    help='length of lists for testing')
parser.add_argument('--saved_model_path', type=str, default='', 
                    help='path to load previous model')
parser.add_argument('--opt', type=str, default='SGD', 
                    help='SGD or ADAM')
parser.add_argument('--num_letters', type=int, default=26, 
                    help='Number of letters in vocabulary')
parser.add_argument('--checkpoint_style', type=str, default='sim_one', 
                    help='acc or sim_one')
parser.add_argument('--stopping_eta', type=float, default = 0.05,
                    help='If accuracy is within this distance from stopping_criteria, stop training')
parser.add_argument('--fb_bool', type=bool, default = True,
                    help='Whether or not incorporate feedback')
parser.add_argument('--repeat_prob', type=float, default=0.0,
                    help='fraction of training trials to sample with replacement')
parser.add_argument('--bias', type=bool, default=False,
                    help='If true, a bias term is added to the recurrent units and output') 
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('--alpha_s', type=float, default=1.0,
                    help='If lower than 1, continuous RNN is implemented')
parser.add_argument('--stateful', type=bool, default=False,
                    help='If true, train stateful RNN where hidden state from previous trial is used to initialize next trial.')
parser.add_argument('--delay_start', type=int, default=0,
                    help='Delay until start of trial')
parser.add_argument('--delay_middle', type=int, default=0,
                    help='Delay before recall')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='L2 reg')
parser.add_argument('--double_trial', type=bool, default=False,
                    help='If true, one trial contains two lists.')
parser.add_argument('--num_letters_test', type=int, default=12,
                    help='Number of letters during testing.')
parser.add_argument('--run_number', type=str, default='',
                    help='Used when performing multiple runs with the same hyperparameters')
parser.add_argument('--storage_capacity', type=int, default=3,
                    help='Number of states that the RNN can store')
parser.add_argument('--test_mode', type=str, default='no_grad',
                    help='If set to no_grad, then weights are not updated during testing.\
                    If set to grad, then weights are updated during testing.')

args = parser.parse_args()

test_path_dict = f'test_set/test_lists_cleaned_{args.num_letters_test}.pkl'
test_path_set  = f'test_set/test_lists_cleaned_{args.num_letters_test}_set.pkl'
test_path_dict_protrusions = f'test_set/test_lists_cleaned_{args.num_letters_test}_protrusions.pkl'

def train_loop(settings, checkpoint_epoch=10000):

    run = wandb.init(project="serial_recall_RNNs", config=settings)
    run.name = f'EM_opt_{run.config.opt}_{run.config.run_number}_dt_{run.config.double_trial}'
    run_folder = save_model_path + wandb.run.name
    os.makedirs(run_folder)

    wandb.config['input_size'] = wandb.config['output_size'] = wandb.config['num_letters'] + 2

    loss_list = []

    best_acc = 0.0
    patience = 0
    human_acc_diff = 100.0

    trainData = OneHotLetters(wandb.config['max_length'], wandb.config['num_cycles']+1, 
                                wandb.config['test_path_set'], wandb.config['output_size'],
                                wandb.config['batch_size'], 
                                num_letters=wandb.config['num_letters'], 
                                repeat_prob=wandb.config['repeat_prob'], 
                                double_trial=wandb.config['double_trial'], 
                                delay_start=wandb.config['delay_start'], 
                                delay_middle=wandb.config['delay_middle'])

    train_dataloader= DataLoader(trainData, batch_size=wandb.config['train_batch'], shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
 
    model = RNN_one_layer_EM(wandb.config['input_size'], wandb.config['hs'], wandb.config['output_size'], wandb.config['fb_bool'], 
    wandb.config['bias'], wandb.config['nonlin'], wandb.config['noise_std'], wandb.config['alpha_s'], wandb.config['storage_capacity'])

    model = model.to(device)

    wandb.watch(model, log='all', log_freq=checkpoint_epoch)

    if wandb.config['opt'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config['lr'], 
        weight_decay=wandb.config['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'], 
        weight_decay=wandb.config['weight_decay'])

    if len(wandb.config['saved_model_path']) > 0:
        stored_model = torch.load(save_model_path + wandb.config['saved_model_path'])
        model.load_state_dict(stored_model['model_state_dict'])
        optimizer.load_state_dict(stored_model['optimizer_state_dict'])

    model.train()
    loss_per_1000 = 0.0
    
    for batch_idx, (X, y) in enumerate(train_dataloader):
        
        list_length = batch_idx%wandb.config['max_length'] + 1
        
        EM_encode_timesteps = encoding_policy_presentation_recall(list_length,
                                            wandb.config['delay_start'], wandb.config['delay_middle'])
        
        # define timesteps where retrieval mechanism is on 
        recall_start_time = wandb.config['delay_start'] + list_length + wandb.config['delay_middle']
        recall_end_time = recall_start_time + list_length
    
        X = X.to(device)
        y = y.to(device)
        
        # Compute prediction and loss
        if batch_idx==0 or wandb.config['stateful']==False:
            # init initial states
            y0, h0, i0 = model.init_states(1, device, 
            0.5)
        # if stateful is true and not on the first trial
        else:
            y0 = y_hat.detach()
            h0 = h.detach()
            i0 = i.detach()

        loss = 0.0

        # run RNN and compute loss
        for timestep in range(X.shape[1]):
            
            if timestep in EM_encode_timesteps:
                model.encoding_on()
            else:
                model.encoding_off()
                
            # only retrieve from EM during recall period
            if timestep == recall_start_time:
                model.retrieval_on()
            if timestep == recall_end_time:
                model.retrieval_off()

            # initial feedback 
            if timestep == 0:
                y_hat, h, i = model(X[:, timestep, :], h0, y0, i0, device)
            else:
                y_hat, h, i = model(X[:, timestep, :], h, y[:, timestep-1, :], i, device)

            loss += loss_fn(y_hat, y[:, timestep, :])
            loss_per_1000 += loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # ensure that norm of all gradients falls under clip_factor
        if wandb.config['clipping']: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), wandb.config['clip_factor'])

        optimizer.step()

        # print model loss every 1000 trials 
        if batch_idx % 1000 == 0 and batch_idx != 0:

            print("Batch number: ", batch_idx)
            loss_per_1000 /= 1000
            if batch_idx != 0:
                loss_list.append(round(loss_per_1000.item(), 5))
            print("Loss: ", round(loss_per_1000.item(), 5))
            wandb.log({'loss: ': round(loss_per_1000.item(), 5)})
            loss_per_1000 = 0.0

        # checkpoint 
        if batch_idx % checkpoint_epoch == 0 and batch_idx != 0:

            if wandb.config['checkpoint_style'] == 'acc':
                # check accuracy every checkpoint_epoch trials and save model
                run_test = run_test_trials_EM(model, wandb.config['h0_init_val'])
                accuracy_dict, mean_acc = checkpoint_acc(run_test)
                wandb.log(accuracy_dict)

                print("Accuracy: ", accuracy_dict[str(wandb.config['test_list_length'])])
                wandb.log({'mean_acc': mean_acc})

                if min(accuracy_dict.values()) > wandb.config['min_acc'] and mean_acc > best_acc:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': accuracy_dict
                        }, run_folder + '/model_best.pth')
                    patience = 0.0
                    best_acc = mean_acc
                else:
                    patience += 1

                # Stopping criteria) impose a minimum accuracy threshold before ending training 
                # to ensure that training isn't stopped early on
                if (patience > wandb.config['patience_limit'] and best_acc > 0.5) or torch.isnan(loss):
                    print("Done training")
                    break

            elif wandb.config['checkpoint_style'] == 'sim_one':

                sim_one = simulation_one(model, optimizer, wandb.config['max_length'], wandb.config['h0_init_val'],
                wandb.config['test_list_length'])
                sim_one_updated = checkpoint_sim_one(sim_one)
                sim_one_updated.log_metrics(wandb)
                accuracy = round(sim_one_updated.ppr_test.item(),5)
                print("Accuracy: ", accuracy)
                wandb.log({'accuracy': accuracy})
                
                # if model comes closer to human accuracy, save 
                if np.abs(accuracy - wandb.config['stopping_criteria']) < human_acc_diff:
                    human_acc_diff = np.abs(accuracy - wandb.config['stopping_criteria']) 
                    torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy
                    }, run_folder + '/model_human_temp.pth')
                    patience = 0
                else:
                    patience += 1 

                if np.abs(sim_one_updated.ppr_test.item() - wandb.config['stopping_criteria']) \
                        <= wandb.config['stopping_eta'] or patience > wandb.config['patience_limit']:
                    print("Done training")
                    torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy
                    }, run_folder + '/model_human.pth')
                    sim_one.figure_six_plot(wandb)
                    sim_one.figure_seven_plot(wandb)
                    break

        # save model every 100,000 trials 
        if batch_idx % (checkpoint_epoch * 10) == 0 and batch_idx!=0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy
                }, run_folder + '/model_' + str(batch_idx) + '.pth')

def checkpoint_acc(run_test):
    run_test.checkpoint(wandb.config, device)
    return run_test.accuracy_dict, run_test.accuracy_list

def checkpoint_sim_one(sim_one):
    for ll in range(4, wandb.config['max_length']+1, 1):
        sim_one.run_model(device, ll, wandb.config)
    return sim_one

# set settings
settings = {
    'max_length' : 9, 
    'test_list_length': args.test_list_length, 
    'num_cycles': 500000,
    'train_batch': 1,
    'num_letters': args.num_letters,
    'hs': args.hs,
    'lr': args.lr,
    'stopping_criteria': args.stopping_criteria,
    'opt': args.opt,
    'nonlin': args.nonlin,
    'clipping': args.clipping, 
    'clip_factor': args.clip_factor,
    'h0_init_val': args.h0_init_val, 
    'noise_std': args.noise_std,
    'test_path_lists': test_path_dict,
    'test_path_set': test_path_set,
    'test_path_protrusions': test_path_dict_protrusions,
    'saved_model_path': args.saved_model_path,
    'patience_limit': 50, # number of checkpoints
    'checkpoint_style': args.checkpoint_style,
    'stopping_eta': args.stopping_eta,
    'fb_bool': args.fb_bool,
    'repeat_prob': args.repeat_prob,
    'bias': args.bias,
    'stateful': args.stateful, 
    'batch_size': args.batch_size,
    'alpha_s': args.alpha_s,
    'delay_start': args.delay_start,
    'delay_middle': args.delay_middle,
    'weight_decay': args.weight_decay, 
    'double_trial': args.double_trial, 
    'grad_mode' : args.test_mode,
    'num_letters_test': args.num_letters_test,
    'run_number': args.run_number,
    'storage_capacity': args.storage_capacity
}

train_loop(settings)





    








    



