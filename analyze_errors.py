import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class analyze_errors():
    
    '''
    Analyzes ISR errors. 
    '''
    
    def __init__(self):
        
        self.intrusions = np.zeros(self.list_length)
        self.immediate_intrusions = np.zeros(self.list_length)
        self.output_protrusions = np.zeros(self.list_length)
        self.transpositions = np.zeros(self.list_length)
        self.repetitions_out_pos = np.zeros(self.list_length)
        self.omissions_out_pos = np.zeros(self.list_length)
        self.repetitions_inp_pos = np.zeros(self.list_length)
        self.omissions_inp_pos = np.zeros(self.list_length)
        self.frac_errors = np.zeros(self.list_length)
        self.ppr = 0 
        self.transposition_gradient = {}
        for i in range(self.list_length):
            self.transposition_gradient[i+1] = []
      
    def compute_errors(self, correct_item, recalled_item, recall_position):

        if recalled_item == correct_item:
            self.transposition_gradient[recall_position+1].append(recall_position+1)
        else:
            self.frac_errors[recall_position] += 1
            # if recalled item is an omission
            if recalled_item == self.omission:
                self.omissions_out_pos[recall_position] += 1
            elif recalled_item not in self.current_list:
                self.intrusions[recall_position] += 1
                # check if there is a previously recalled list
                if len(self.previously_recalled_list) != 0:
                    # check if recalled item is an immediate intrusion 
                    if recalled_item in self.previously_recalled_list:
                        self.immediate_intrusions[recall_position] += 1
                        # check if immediate intrusion is an output protrusion 
                        ii_pos = np.argwhere(recalled_item == np.array(self.previously_recalled_list))[0][0]
                        if ii_pos == recall_position:
                            self.output_protrusions[recall_position] += 1
            else:
                # if there is a previous instance of that item in the list
                # then the item is a repetition error
                if recalled_item in self.recalled_list: 
                    self.repetitions_out_pos[recall_position] += 1
                    repetition_input_idx = np.argwhere(self.current_list == recalled_item)[0][0]
                    self.repetitions_inp_pos[repetition_input_idx] += 1
                self.transpositions[recall_position] += 1
                recalled_item_pos = np.argwhere(self.current_list == recalled_item)[0][0]
                self.transposition_gradient[recall_position+1].append(recalled_item_pos+1)
                
    def compute_input_omissions(self):
        
        '''
        Computes omissions based on input position. 
        Needed a separate function for this because doing so requires
        the entire list to have been recalled. 
        '''
        
        for pos_inp, item in enumerate(self.current_list):
            if item not in self.recalled_list:
                self.omissions_inp_pos[pos_inp] += 1
                 
    def compute_acc(self):
        
        if np.array_equal(self.recalled_list, self.current_list):
            self.ppr += 1
            
    def compute_num_trials(self):
        
        self.num_trials = len(self.transposition_gradient[1])
            
    def plot_transposition_gradient(self, save_name):
        
        fig, ax = plt.subplots(1,6, sharex=False, sharey=True, figsize=(12,8))
        sns.despine()
        for i in range(self.list_length): 
            ax[i].hist(x=self.transposition_gradient[i+1], bins=np.arange(1, self.list_length+2, 1), 
            density=True, color=(0,0,0), rwidth=.8)
            ax[i].set_xlabel(i+1, fontsize=18)
            ax[i].set_xticks([])
            ax[i].set_yticks([0, .25, .5, .75, 1])
            ax[0].tick_params(axis='y', which='major', labelsize=18)
            if i > 0:
                ax[i].get_yaxis().set_visible(False)
                sns.despine(left=True, ax=ax[i])
        plt.subplots_adjust(wspace=.1, hspace=.1)
        plt.text(-15, -.1, 'Position', fontsize=24)
        plt.show()
        fig.savefig(self.save_folder + save_name, dpi=300, bbox_inches='tight')
        
    def plot_errors_inp_out(self, error_type, frac_responses, save_name):
        
        '''
        :param str error_type: rep for repetition errors, omi for omission errors 
        :param bool frac_responses: if true, plots error type as a fraction of all responses. 
        If false, plots error type as a function of overall errors. 
        :param str save_name: where to save figure
        '''
        
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        sns.despine()
        
        if error_type == 'rep':
            error_inp_pos = self.repetitions_inp_pos
            error_out_pos = self.repetitions_out_pos
        elif error_type == 'omi':
            error_inp_pos = self.omissions_inp_pos
            error_out_pos = self.omissions_out_pos
        else:
            print("Error type is not valid, must be either rep or omi")
            return 0 
            
        if frac_responses:
            repetition_rate_inp = error_inp_pos / self.num_trials
            repetition_rate_out = error_out_pos / self.num_trials
            ylabel_str = "Fraction of responses"
        else:
            repetition_rate_inp = error_inp_pos / self.frac_errors
            repetition_rate_out = error_out_pos / self.frac_errors
            ylabel_str = "Fraction of errors"
            
        ax[0].plot(np.arange(1,self.list_length+1, 1), repetition_rate_inp, marker='o')
        ax[0].set_ylabel(ylabel_str, fontsize=14)
        ax[0].set_xlabel("Serial positions (Input)", fontsize=14)
        ax[1].plot(np.arange(1,self.list_length+1, 1), repetition_rate_out, marker='o')
        ax[1].set_xlabel("Serial positions (Output)", fontsize=14)
        fig.show()
        fig.savefig(self.save_folder + save_name, dpi=300)
        
        
        
    
    