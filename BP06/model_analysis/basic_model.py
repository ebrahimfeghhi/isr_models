'''
This class is a parent class that contains basic functions for loading and running BP06.
'''

import torch
import sys
sys.path.append('/home3/ebrahim/isr/isr_model_review/BP06/')
from run_test_trials import run_test_trials
from simulation_one import simulation_one
from RNNcell import RNN_one_layer
import wandb
device = torch.device("cpu")

class load_model_basic():

    def __init__(self, base):
        
        self.base = base

    def load_model(self, wandbPath, modelPath):

        api = wandb.Api()
        run = api.run(wandbPath)
        self.config = run.config

        # modify paths
        self.config['test_path_lists'] = self.base + self.config['test_path_lists']
        self.config['test_path_set'] = self.base + self.config['test_path_set']
        self.config['test_path_protrusions'] = self.base + self.config['test_path_protrusions']

        saved_info = torch.load(self.base + 'saved_models/' +  modelPath  + '/model_human.pth')
        self.model = RNN_one_layer(self.config['input_size'], self.config['hs'], self.config['output_size'], noise_std=self.config['noise_std'],
                                feedback_bool=True, bias=False, alpha_s=self.config['alpha_s'])
        self.model.load_state_dict(saved_info['model_state_dict'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.optimizer.load_state_dict(saved_info['optimizer_state_dict'])

    def run_model(self, ll):

        rtt = run_test_trials(self.model, self.optimizer, 0.5)
        rtt.run_model(device, ll, self.config, self.base)
        X_test = rtt.X_test_all
        self.h = rtt.h_stacked.numpy()
        self.y_recall_probs = rtt.y_hat_all
        self.y_recall = rtt.y_hat_recall

    def run_simulation_one(self, ll):

        self.sim_one = simulation_one(self.model, self.optimizer, self.config['max_length'], self.config['h0_init_val'],
                self.config['test_list_length'])

        self.sim_one.run_model(device, ll, self.config)