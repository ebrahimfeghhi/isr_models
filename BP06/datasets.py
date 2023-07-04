from cgi import test
import torch 
from torch.utils.data import Dataset
import numpy as np
import pickle

class OneHotLetters_test(Dataset):

    def __init__(self, test_path, list_length, num_classes, num_letters=26, 
    dict_key='', delay_start=0, delay_middle=0, double_trial=True, pr_recall_l2 = 1.0):
        
        self.list_length = list_length
        self.num_classes = num_classes
        self.num_letters = num_letters
        self.delay_start = delay_start
        self.delay_middle = delay_middle
        self.double_trial = double_trial
        self.pr_recall_l2 = pr_recall_l2
   
        with open(test_path, 'rb') as f:
            self.test_data = pickle.load(f)
            
        '''
        Load test trials for a given list length 
        :param list test_path: path to test lists
        :param int list_length: desired list length used for testing
        :param int num_letters: number of letters in vocabulary
        :param str dict_key: dict key to access test_data. If empty, will use list_length as key
        '''
    
        if len(dict_key) == 0:
            dict_key = str(list_length)

        X_int = np.stack(self.test_data[dict_key])
        
        self.num_trials = X_int.shape[0]

        delay_start_t = np.ones((self.num_trials, delay_start)) * (num_letters+1)
        delay_middle_t = np.ones((self.num_trials, delay_middle)) * (num_letters+1)

        recall_duration = list_length + 1

        recall_cue = np.ones((self.num_trials, recall_duration)) * num_letters 

        self.X = torch.nn.functional.one_hot(torch.from_numpy(
                np.hstack((delay_start_t, X_int, delay_middle_t, recall_cue))).to(torch.long), num_classes=num_classes)

        end_of_list_cue = np.ones((self.num_trials, 1)) * num_letters
        y_int = torch.from_numpy(np.hstack((delay_start_t, X_int, delay_middle_t, 
                                            X_int, end_of_list_cue))).to(torch.long)
        self.y = torch.nn.functional.one_hot(y_int, num_classes=num_classes)

    def __len__(self):

        return self.X.shape[0]
    
    def construct_double_trial(self, idx):
        
        # if at the last trial, make the double trial the last
        # trial followed by the first trial
        if idx == self.num_trials-1:
            l2_idx = 0 
        else:
            l2_idx = idx + 1
        
        rng = np.random.default_rng()
        
        X1 = self.X[idx]
        X2 = self.X[l2_idx]
        y1 = self.y[idx]
        y2 = self.y[l2_idx]
        
        presentation_phase = self.delay_start + self.list_length
        X1_presentation = X1[:presentation_phase]
        y1_presentation = y1[:presentation_phase]
        X2_presentation = X2[:presentation_phase]
        y2_presentation = y2[:presentation_phase]
        
        select_trial_for_recall = rng.integers(0,100)/100
        if select_trial_for_recall < self.pr_recall_l2:
            context_cue = self.num_letters + 2
            X_recall = X2[presentation_phase:]
            y_recall = y2[presentation_phase:]
        else:
            context_cue = self.num_letters + 3
            X_recall = X1[presentation_phase:]
            y_recall = y1[presentation_phase:]
            
        context_cue = torch.nn.functional.one_hot(torch.from_numpy(np.asarray(context_cue)), 
                                                  num_classes=self.num_classes).unsqueeze(0)
            
        X = torch.cat((X1_presentation, X2_presentation, context_cue, X_recall), axis=0)
        y = torch.cat((y1_presentation, y2_presentation, context_cue, y_recall), axis=0)
        
        return X, y
        
    def __getitem__(self, idx):
        
        if self.double_trial:
            X, y = self.construct_double_trial(idx)
        else:
            X = self.X[idx]
            y = self.y[idx]
            
        return X.to(torch.float32), y.to(torch.float32)

class OneHotLetters(Dataset):
  
    def __init__(self, max_length, num_cycles, test_path, num_classes, batch_size=1, num_letters=26, 
    repeat_prob=0.0, delay_start=2, delay_middle=1, double_trial=True, pr_recall_l2 = 0.8):

        """ 
        Initialize class to generate letters, represented as one hot vectors in 26 dimensional space. 
        :param int max_length: maximum number of letters 
        :param int num_cycles: number of cycles (1 cycle = set of lists of length 1,...,max_length)
        :param int num_classes: number of classes 
        :param int batch_size: size of each batch
        :param int num_letters: number of letters in vocabulary 
        :param str test_path: path to test_list (type should be set for quick look up)
        :param float repeat_prob: fraction of trials to sample with repetition
        :param int delay_start: how much delay before trial starts 
        :param int delay_middle: how much delay between retrieval and recall 
        :param bool double_trial: if true, one list contains two trials. BP06 only needs to 
        recall one list, depending on what the context cue is. 
        :param float pr_recall_l2: the probability that the context cue signals recall of list 2
        """ 
        
        self.max_length = max_length
        self.num_letters = num_letters
        self.num_cycles = num_cycles
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.storage = []
        self.repeat_prob = repeat_prob
        self.delay_start = delay_start
        self.delay_middle = delay_middle
        self.list_length = 0 
        self.double_trial = double_trial
        self.pr_recall_l2 = pr_recall_l2
        
        with open(test_path, 'rb') as f:
            self.test_data = pickle.load(f)

    def __len__(self):

        return self.num_cycles * self.max_length

    def construct_trial(self): 

        '''
        Generates a training example
        '''

        rng = np.random.default_rng()

        delay_start = np.ones(self.delay_start) * (self.num_letters+1)
        delay_middle = np.ones(self.delay_middle) * (self.num_letters+1)

        letters = rng.choice(self.num_letters, self.list_length, replace=False) 
            
        # ensure train and test are not overlapping
        if self.list_length > 1: 
            while tuple(letters) in self.test_data[str(self.list_length)]:
                rng = np.random.default_rng()
                letters = rng.choice(self.num_letters, self.list_length, replace=False) 
                
        recall_cue = np.ones(self.list_length+1) * self.num_letters 

        X = torch.nn.functional.one_hot(torch.from_numpy(
            np.hstack((delay_start, letters, delay_middle, recall_cue))).to(torch.long),
        num_classes=self.num_classes)

        # output is letters during letter presentation
        # letters again after recall cue
        # and finally end of list cue 
        y = torch.from_numpy(np.hstack((delay_start, letters, delay_middle,
        letters, self.num_letters))).to(torch.long)
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes)

        return X, y
    
    def construct_double_trial(self):
        
        '''
        Creates a trial where two lists are presented, 
        followed by a context cue indicating which list to recall. 
        '''
        
        rng = np.random.default_rng()
        
        X1, y1 = self.construct_trial()
        X2, y2 = self.construct_trial()
        
        presentation_phase = self.delay_start + self.list_length
        X1_presentation = X1[:presentation_phase]
        y1_presentation = y1[:presentation_phase]
        X2_presentation = X2[:presentation_phase]
        y2_presentation = y2[:presentation_phase]
        
        select_trial_for_recall = rng.integers(0,100)/100
        if select_trial_for_recall < self.pr_recall_l2:
            context_cue = self.num_letters + 2
            X_recall = X2[presentation_phase:]
            y_recall = y2[presentation_phase:]
        else:
            context_cue = self.num_letters + 3
            X_recall = X1[presentation_phase:]
            y_recall = y1[presentation_phase:]
            
        context_cue = torch.nn.functional.one_hot(torch.from_numpy(np.asarray(context_cue)), 
                                                  num_classes=self.num_classes).unsqueeze(0)
            
        X = torch.cat((X1_presentation, X2_presentation, context_cue, X_recall), axis=0)
        y = torch.cat((y1_presentation, y2_presentation, context_cue, y_recall), axis=0)
        
        return X, y
        
    def __getitem__(self, idx):
        
        rng = np.random.default_rng()

        # every new batch, increment the list_length 
        # once the list length exceeds the max length, return back to 1 
        if idx % self.batch_size == 0: 
            self.list_length += 1
            if self.list_length > self.max_length:
                self.list_length = 1
                
        if self.double_trial:
            X, y = self.construct_double_trial()
        else:
            X, y = self.construct_trial()

        return X.to(torch.float32), y.to(torch.float32) 
    
    
