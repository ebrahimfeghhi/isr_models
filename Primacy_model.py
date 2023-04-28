import numpy as np
import matplotlib.pyplot as plt 
from analyze_errors import analyze_errors

class primacy_model(analyze_errors):

    def __init__(self, params_dict):

        '''
        :param float P: peak value of undecayed primacy gradient 
        :param float M: sd of noise added to forward activation
        :param float T: output threshold
        :param float N: sd of selection noise 
        :param float blank: period of time from end of one item to onset of the next
        :param float R: covert rehearsal rate (items/sec)
        :param float item_presentation_rate: length (sec) that each item is presented
        :param int vocab_size_PM: number of items in vocabulary 
        :param float input_strength: how strongly each input is activated upon presentation 
        :param int list_length: length of each list 
        :param float output_time: time it takes to output a letter
        '''
        
        self.P = params_dict['P']
        self.M = params_dict['M']
        self.T = params_dict['T']
        self.D = params_dict['D']
        self.N = params_dict['N']
        self.s = params_dict['P']
        self.blank_period = params_dict['blank']
        self.R = params_dict['R']
        self.ipr = params_dict['item_presentation_rate']
        self.vocab_size_PM = params_dict['vocab_size_PM']
        self.input_strength = params_dict['input_strength']
        self.list_length = params_dict['list_length']
        self.output_time = params_dict['output_time']
        self.dt = params_dict['dt']
        self.save_folder = params_dict['save_folder']
        self.R_s = params_dict['R_s']
        
         # inter-onset interval 
        self.IOI = self.ipr + self.blank_period
        self.C = np.floor(max(0, self.R * (self.IOI - .2))) # number of cumulative rehearsals 
        self.omission = -1 
        self.intrusions = 0

        # define number of euler updates for item presentation, ioi, and recalled
        self.euler_updates_item_presentation = int(self.ipr/self.dt)
        self.euler_updates_blank = int(self.blank_period/self.dt)
        self.euler_updates_recalled = int(self.output_time/self.dt)
        
        super().__init__() # init analyze errors class 
        
    def present_list(self, presented_items):

        self.item_activations = np.zeros(self.vocab_size_PM)
        self.start_marker = self.s
        self.L = len(presented_items)
        self.current_list = presented_items

        for pos, item in enumerate(presented_items):
            item_inputs = np.zeros(self.vocab_size_PM)
            item_inputs[item] = self.input_strength
            self.activation_dynamics(False, item_inputs, pos)

    def recall_list(self):

        self.recalled_list= []

        for i in range(self.list_length):
            
            correct_item = self.current_list[i]
            
            # add selection noise
            item_act_noisy = self.item_activations + \
            np.random.default_rng().normal(0, self.N, self.vocab_size_PM)

            # retrieve strongest activated item 
            selected_item = np.argmax(item_act_noisy)
            selected_item_act = np.max(item_act_noisy)       

            # add noise to selected item before comparing to output threshold 
            selected_item_act += np.random.default_rng().normal(0, self.M, 1)[0]
            
            # check if activation of selected item is greater than the threshold
            if selected_item_act >= self.T:
                recalled_item = selected_item
                
                # exponentially decay item activation 
                self.item_activations[selected_item] *= np.exp(-self.R_s)
                
            else:
                recalled_item = self.omission
                
            self.compute_errors(correct_item, recalled_item, i)
            self.recalled_list.append(recalled_item)
            self.activation_dynamics(recall_mode=True, item_inputs=np.zeros(self.vocab_size_PM))
 
    def activation_dynamics(self, recall_mode, item_inputs, position=None):

        # incorporate exponential decay for recall phase
        if recall_mode: 
            for i in range(self.euler_updates_recalled):
                self.item_activations += -self.D * self.item_activations * self.dt
        else: 
            
            n = np.sum(self.item_activations>0) # number of items presented 
            
            for i in range(self.euler_updates_item_presentation):

                # if the entire list can be rehearsed, then remove exponential decay
                # otherwise incorporate exp decay
                if position <= self.C:
                    exponential_decay = np.zeros(self.vocab_size_PM)
                    exponential_decay_sm = 0
                else:
                    exponential_decay = -self.D * self.item_activations
                    exponential_decay_sm = -self.D * self.start_marker # decay for start marker 
                
                A = self.start_marker*(1-n/self.P)
                self.item_activations += (exponential_decay + (A-self.item_activations)*item_inputs)*self.dt 
                self.start_marker += exponential_decay_sm*self.dt

            # incorporate decay effects for inter-item interval 
            # only if cumulative rehearsals are no longer possible
            if position > self.C:
                for i in range(self.euler_updates_blank):
                    self.item_activations += -self.D * self.item_activations * self.dt
                    self.start_marker += self.start_marker*-self.D*self.dt


    def simulate_trials_PM(self, num_trials):
        
        self.num_trials = num_trials

        self.presented_list_storage = np.zeros((num_trials, self.list_length))
        self.recalled_list_storage = np.zeros((num_trials, self.list_length))

        for i in range(num_trials):

            vocab = np.arange(self.vocab_size_PM)

            current_list= np.random.default_rng().choice(vocab, self.list_length, replace=False)

            self.presented_list_storage[i] = current_list

            self.present_list(current_list)
            self.recall_list()
            
            self.recalled_list_storage[i] = self.recalled_list
            
            self.compute_acc()
            self.compute_input_omissions()
            
    def softmax(self, activations):
    
        return np.exp(activations)/np.sum(np.exp(activations))