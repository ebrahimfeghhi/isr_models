import torch 
import numpy as np
from datasets import OneHotLetters, OneHotLetters_test
from torch.utils.data import DataLoader

class run_test_trials():

    def __init__(self, model, optimizer, h0_init_val):

        self.model = model
        self.optimizer = optimizer 
        self.h0_init_val = h0_init_val
        self.accuracy_dict = {}
        self.accuracy_list = []
        self.hidden_activity_dict = {}

    def run_single_trial(self, X_test, y_test, h, i, y_hat_onehot, hebb, device, config):

        if config['grad_mode'] == 'no_grad':

            with torch.no_grad():

                y_hat_single_trial = []
                h_single_trial = []
                i_single_trial = []

                h_single_trial.append(h)
                i_single_trial.append(i)

                for timestep in range(X_test.shape[1]):
                                
                    y_hat, h, hebb, i = self.model(X_test[:, timestep, :], h, y_hat_onehot, hebb, i, device)
                    
                    y_hat = torch.softmax(y_hat, dim=1)
                    y_hat_onehot = torch.nn.functional.one_hot(torch.argmax(y_hat, dim=1), 
                                    num_classes=y_hat.shape[1]).to(torch.float32)

                    y_hat_single_trial.append(y_hat.squeeze())
                    
                    h_single_trial.append(h.detach())
                    i_single_trial.append(i.detach())

        if config['grad_mode'] == 'grad':

            loss = 0.0

            y_hat_single_trial = []
            h_single_trial = []
            i_single_trial = []

            h_single_trial.append(h)
            i_single_trial.append(i)

            # iterate to 2nd to last input (b/c ignoring end of list cue)
            for timestep in range(X_test.shape[1]):
                                
                y_hat, h, hebb, i = self.model(X_test[:, timestep, :], h, y_hat_onehot, hebb, i, device)
                
                y_hat = torch.softmax(y_hat, dim=1)
                y_hat_onehot = torch.nn.functional.one_hot(torch.argmax(y_hat, dim=1), 
                                num_classes=y_hat.shape[1]).to(torch.float32)

                y_hat_single_trial.append(y_hat.squeeze())
                
                h_single_trial.append(h.detach())
                i_single_trial.append(i.detach())

                loss += self.loss_fn(y_hat, y_test[:, timestep, :])
        
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # ensure that norm of all gradients falls under clip_factor
            if config['clipping']: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['clip_factor'])

            self.optimizer.step()

        return y_hat_single_trial, h_single_trial, i_single_trial, h, i, y_hat_onehot, hebb


    def run_model(self, device, test_list_length, config, base, dataloader=None, dict_key='', 
    protrusions=False, storage_trials=False, shuffle=True):
        
        # total delay
        td = config['delay_start'] + config['delay_middle']
        if dataloader==None:
            if protrusions:
                dataset = OneHotLetters_test(base + config['test_path_protrusions'], test_list_length, config['output_size'],
                config['num_letters'], dict_key, delay_start=config['delay_start'], delay_middle=config['delay_middle'])
            else:
                dataset = OneHotLetters_test(base + config['test_path_lists'], test_list_length, config['output_size'],
                config['num_letters'], dict_key, delay_start=config['delay_start'], delay_middle=config['delay_middle'], 
                delay_storage=config['delay_storage'])

            # We'll keep the batch size to 1 for testing
            dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.model.to(device)

        y_hat_list = []
        y_test_list = []
        X_test_list = []
        h_list = []
        i_list = []

        # generate model_outputs
        for batch_idx, (X_test, y_test) in enumerate(dataloader):

            X_test = X_test.to(device)
            y_test = y_test.to(device)

            y_test_list.append(y_test)
            X_test_list.append(X_test)

            # Compute prediction and loss
            if batch_idx==0 or config['stateful']==False:
                y_hat_onehot, h, hebb, i = self.model.init_states(config['train_batch'], device, 
                config['h0_init_val'])
            else:
                # if stateful is true and not on the first trial, then initialize hidden state
                # with final hidden state from previous trial 
                _, _, _, _ = self.model.init_states(config['train_batch'], device, 
                config['h0_init_val'])
                y_hat_onehot = y_hat_onehot.detach()
                h = h.detach()
                hebb = hebb.detach()
                i = i.detach()

            self.model.eval()

            y_hat_single_trial, h_single_trial, i_single_trial, h, i, y_hat_onehot, hebb = \
            self.run_single_trial(X_test, y_test, h, i, y_hat_onehot, hebb, device, config)

            y_hat_list.append(torch.stack(y_hat_single_trial))
            h_list.append(torch.stack(h_single_trial))
            i_list.append(torch.stack(i_single_trial))

        self.y_test_all = torch.stack(y_test_list).squeeze()
        self.X_test_all = torch.stack(X_test_list).squeeze()
        self.y_test_recall = self.y_test_all[:, td+test_list_length:td+(test_list_length*2), :].cpu().argmax(2)
        self.y_test = self.y_test_all.cpu().argmax(2)
        self.y_hat_all = torch.stack(y_hat_list).squeeze()
        self.y_hat_recall = self.y_hat_all[:, td+test_list_length:td+(test_list_length*2), :].cpu().argmax(2)
        self.y_hat = self.y_hat_all.cpu().argmax(2)
        self.h_stacked = torch.stack(h_list).squeeze()

    def compute_ppr(self):

        # porportion of lists perfectly recalled
        ppr = torch.all(torch.eq(self.y_hat_recall, self.y_test_recall), dim=1).sum() \
        / self.y_hat_recall.shape[0]

        return ppr 

    def compute_ppr_all(self):
        
        # same as above compute_ppr, but evaluates accuracy across entire list 
        ppr = torch.all(torch.eq(self.y_hat, self.y_test), dim=1).sum() \
        / self.y_hat.shape[0]

        return ppr 

    def correct_trials(self):

        correct_trial_indices = torch.nonzero(torch.all(torch.eq(self.y_hat_recall, self.y_test_recall), dim=1))
        return correct_trial_indices

    def checkpoint(self, config, device, base=''):

        for ll in range(2, config['max_length']+1):
            self.run_model(device, ll, config, base)
            accuracy = self.compute_ppr()
            self.accuracy_list.append(accuracy.item())
            self.accuracy_dict[str(ll)] = round(accuracy.item(),5)
            self.hidden_activity_dict[str(ll)] = self.h_stacked


        