import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
torch.set_num_threads(8)
import seaborn as sns
from run_test_trials import run_test_trials

class simulation_one(run_test_trials):

    def __init__(self, model, optimizer, max_length, h0_init_val, test_length, base=''):

        '''
        @param model: PyTorch model 
        @param optimizer: PyTorch optimizer
        @param max_length: longest list the model is tested on
        @param max_length: longest list the model is tested on 
        @param h0_init_val: init. firing rates at start of trial
        @param test_length: lenght of lists used for generating figures and other metrics
        '''
        self.model = model
        self.optimizer = optimizer
        self.max_length = max_length
        self.h0_init_val = h0_init_val
        self.test_length = test_length # list length used to compute metrics and create figures 
        self.met_accuracy = False
        self.ppr_list = [] # Proportion perfectly recalled
        self.ppr_test = 0 # ppr on lists of length six 
        self.list_length = []
        self.ARD = 0 # Average repetition distance
        self.R_T = 0 # Fraction of transposition errors that are repetitions 
        self.relative_error_ratio = 0 # Ratio of relative errors (RE)
        self.base = base

    def run_model(self, device, test_list_length, config):

        super().run_model(device, test_list_length, config, self.base)

        ppr = super().compute_ppr()

        if test_list_length == self.test_length:            
            transpositions, ARD, R_T, relative_error_ratio, intrusion_error_ratio \
             = self.transposition_matrix(test_list_length)
            self.immediate_intrusions, self.protrusions = self.compute_ii_and_p(config, device, test_list_length, self.base)
            self.ARD = ARD
            self.R_T = R_T
            self.ppr_test = ppr
            self.transpositions_test = transpositions
            self.relative_error_ratio = relative_error_ratio
            self.intrusion_error_ratio = intrusion_error_ratio
            self.ppr_storage = self.compute_ppr_storage_trials(config, device, test_list_length, self.base)
        
        self.ppr_list.append(ppr)
        self.list_length.append(test_list_length)

    def transposition_matrix(self, test_list_length):

        # records index where model output at a given timestep and trial
        # equals target output
        transpositions = np.empty((self.y_test_recall.shape[1], 
        self.y_hat_recall.shape[0])) 
        repetition_errors = 0
        self.repetition_trials = {}
        self.repetition_distances = []
        self.report_early = 0
        self.fill_in_errors = 0
        self.relative_errors_subset = 0 
        transposition_errors = 0 
        self.relative_errors = 0
        self.non_relative_errors = 0
        intrusion_errors = 0
        intrusion_error_ratio = 0
    
        for trial in range(self.y_hat_recall.shape[0]):
            for timestep in range(test_list_length):

                # indices where target letter at a given timestep equals to recalled 
                # list
                matched_tp = np.argwhere(self.y_test_recall[trial, timestep] == \
                self.y_hat_recall[trial]).numpy().squeeze()

                # if a target item is found more than once in the predicted list
                # increment number of repetition errors by amount of excess repeats
                if matched_tp.size > 1:

                    repetition_errors += matched_tp.size - 1

                    # distance between repeated elements in predictions
                    rep_dist = []
                    for dist in np.ediff1d(matched_tp):
                        self.repetition_distances.append(dist)
                        rep_dist.append(dist)

                    self.repetition_trials[trial]= rep_dist

                # locate position where predicted letter at T = timestep equals target list
                matched_pt = np.argwhere((self.y_hat_recall[trial, timestep] == \
                self.y_test_recall[trial]).numpy())

                # ensure there is a match (i.e. no intrusion error)
                if matched_pt.size == 1:
                    transpositions[timestep, trial] = int(matched_pt) + 1

                    # if recalled letter isn't at correct position, then increment
                    # number of transposition errors
                    if int(matched_pt) != timestep:
                        transposition_errors += 1 
  
                        # relative and fill-in errors can't occur if at last timestep
                        if timestep != test_list_length-1:

                            # Check for a relative error. 
                            # A relative error is when S recalls a letter at the incorrect position, i 
                            # And then at position i+1 recalls the letter that followed that letter in the 
                            # encoding phase. 

                            # Example:
                                # Encoding list: ABCD
                                # Recall list: ACDB

                            # Note how C is recalled at the incorrect position, and then D follows C. 

                            # if the incorrect letter is at the last timestep, a relative error or fill in error is not possible.

                            # timestep where next predicted letter equals to target list
                            # if next predicted letter is not in the list, we have intrusion error and
                            # not an adjacent transposition
                            matched_pt_next = np.argwhere((self.y_hat_recall[trial, timestep+1] == \
                            self.y_test_recall[trial]).numpy())
                            
                            # if next predicted letter is found in target list
                            if matched_pt_next.size==1:

                                # model reported incorrect item at position i, if the next reported item is the item 
                                # that was at position i+1, then we have a relative error 
                                if matched_pt_next - matched_pt == 1:
                                    self.relative_errors += 1

                                # elif the predicted letter at next position does not equal to the target letter
                                # at the next position, we have a non-relative error
                                elif self.y_hat_recall[trial, timestep+1] != self.y_test_recall[trial, timestep+1]:
                                    self.non_relative_errors += 1

                        if timestep < test_list_length - 2:

                            # if predicted letter at T = timestep was the target letter at T = timestep + 1
                            if int(matched_pt) == (timestep + 1):
                                
                                if self.y_hat_recall[trial, timestep+1] == self.y_test_recall[trial, timestep+1]:
                                    pass

                                else:
                                    self.report_early += 1

                                    # check if fill in error occurred 
                                    if self.y_hat_recall[trial, timestep+1] == self.y_test_recall[trial, timestep]:
                                        self.fill_in_errors += 1

                                    # check if fill in error occurred 
                                    if self.y_hat_recall[trial, timestep+1] == self.y_test_recall[trial, timestep+2]:
                                        self.relative_errors_subset += 1

                # if predicted item is not in the target list,
                # then an intrusion error has occurred 
                else:
                    intrusion_errors += 1

        ARD = round(np.mean(self.repetition_distances),3)
        total_errors = transposition_errors+intrusion_errors+repetition_errors
        T_R_errors = transposition_errors+repetition_errors

        # protect against dividing by zero 
        if total_errors == 0:
            self.R_E = 0
            intrusion_error_ratio = 0
        else:
            self.R_E = round(repetition_errors/total_errors, 3)
            intrusion_error_ratio = round(intrusion_errors / (total_errors),3)

        if self.non_relative_errors + self.relative_errors == 0:
            relative_error_ratio = 0
        else:
            relative_error_ratio = round(self.relative_errors/(self.non_relative_errors+self.relative_errors), 3)

        if T_R_errors == 0:
            R_T = 0
        else:
            R_T = round(repetition_errors / T_R_errors, 3)
    
        return transpositions, ARD, R_T, relative_error_ratio, intrusion_error_ratio


    def compute_ii_and_p(self, config, device, ll, base):

        '''
        Computes number of immediate intrusion and protrusion errors. 
        '''

        # modify test path to test on even odd alternating lists 
        super().run_model(device, ll, config, base, dict_key='even_odd', protrusions=True, shuffle=False)

        immediate_intrusions= 0
        protrusions = 0

        for i in range(self.y_hat_recall.shape[0]):

            if i == 0:
                continue

            y_hat_prev = self.y_hat_recall[i-1]
            y_hat_current = self.y_hat_recall[i]
            y_test_current = self.y_test[i]

            # collect indices where y_hat_current equals y_hat_prev
            overlap = np.isin(y_hat_current, y_hat_prev)

            # if there are no matching indices, continue
            if np.sum(overlap) == 0:
                continue 

            matching_indices = np.argwhere(overlap)

            for j in matching_indices:
                # if matching index is not in the presented list, it is an intrusion 
                if y_hat_current[j][0] not in y_test_current: 
                    immediate_intrusions += 1
                    # if intrusion is at the same position on both reports, it is a protrusion 
                    if y_hat_prev[j][0] == y_hat_current[j][0]:
                        protrusions += 1 

        return immediate_intrusions, protrusions

    def compute_ppr_storage_trials(self, config, device, ll, base):
        if config['storage_frac'] == 0:
            return 0
        super().run_model(device, ll, config, base, shuffle=True, storage_trials=True)
        return super().compute_ppr_all()

    def figure_six_plot(self, wandb, displayMode=False):
        fig, ax = plt.subplots(1,1)
        ax.plot(self.list_length, self.ppr_list, marker='o')
        ax.set_xticks(self.list_length)
        ax.set_xlabel("List lengths")
        ax.set_ylabel("Accuracy")
        if displayMode==False:
            wandb.log({'ppr_plot': fig})
            plt.close()
        else:
            plt.show()

    def figure_seven_plot(self, wandb, savePlot=True, displayMode=False):
        
        fig, ax = plt.subplots(1,6, sharex=False, sharey=True, figsize=(12,8))
        sns.despine()
        for i in range(6): 
            ax[i].hist(x = self.transpositions_test[i, :], bins=np.arange(1,self.test_length+2, 1), 
            density=True, color=(0,0,0), rwidth=.8)
            ax[i].set_xlabel(i+1, fontsize=18)
            ax[i].set_xticks([])
            ax[i].set_yticks([0, .25, .5, .75, 1])
            ax[0].tick_params(axis='y', which='major', labelsize=18)
            if i > 0:
                ax[i].get_yaxis().set_visible(False)
                sns.despine(left=True, ax=ax[i])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.text(-15, -.1, 'Position', fontsize=24)
        plt.show()
        if displayMode == False:
            if savePlot:
                wandb.log({'transposition_plot': wandb.Image(fig)})
            plt.close()

    def log_metrics(self, wandb):

        wandb.log({'ARD':self.ARD, 'R_T': self.R_T, 'relative_error_ratios': self.relative_error_ratio, 
        'intrusion_error_ratio': self.intrusion_error_ratio, 'immediate_intrusions': self.immediate_intrusions,
        'protrusions': self.protrusions, 'ppr_storage': self.ppr_storage})



