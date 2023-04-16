from cgi import test
import torch 
from torch.utils.data import Dataset
import numpy as np
from itertools import islice, permutations
import pickle

class OneHotLetters_test(Dataset):

    def __init__(self, test_path, list_length, num_classes, num_letters=26, 
    dict_key='', delay_start=0, delay_middle=0, delay_storage=0):

        with open(test_path, 'rb') as f:
            self.test_data = pickle.load(f)
        '''
        Load test trials for a given list length 
        :param list test_path: path to test lists
        :param int list_length: desired list length used for testing
        :param int num_letters: number of letters in vocabulary
        :param str dict_key: dict key to access test_data. If empty, will use list_length as key
        :param int delay_storage: amount of storage delay 
        '''

        if len(dict_key) == 0:
            dict_key = str(list_length)

        X_int = np.stack(self.test_data[dict_key])

        num_trials = X_int.shape[0]

        delay_start_t = np.ones((num_trials, delay_start)) * (num_letters+1)
        delay_middle_t = np.ones((num_trials, delay_middle)) * (num_letters+1)

        recall_duration = list_length + 1

        recall_cue = np.ones((num_trials, recall_duration)) * num_letters 

        self.X = torch.nn.functional.one_hot(torch.from_numpy(np.hstack((delay_start_t, X_int, delay_middle_t,
        recall_cue))).to(torch.long), num_classes=num_classes)

        end_of_list_cue = np.ones((num_trials, 1)) * num_letters
        y_int = torch.from_numpy(np.hstack((delay_start_t, X_int, delay_middle_t, X_int, end_of_list_cue))).to(torch.long)
        self.y = torch.nn.functional.one_hot(y_int, num_classes=num_classes)

    def __len__(self):

        return self.X.shape[0]
    
    def __getitem__(self, idx):

        return self.X[idx].to(torch.float32), self.y[idx].to(torch.float32)

class OneHotLetters(Dataset):
  
    def __init__(self, max_length, num_cycles, test_path, num_classes, batch_size=1, num_letters=26, 
    repeat_prob=0.0, delay_start=0, delay_middle=0, double_trial=False, storage_frac=0.0, 
    delay_storage=0):

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
        :param bool double_trial: if true, one list contains two trials 
        :param bool storage_frac: if greater than 0, the RNN will be required to recall the previous list
        from it's storage mechanism 
        :param int delay_storage: amount of delay between recalling last item and recalling list again
        If specified, trials will be generated in a reproducible manner. 
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
        self.storage_frac = storage_frac
        self.delay_storage = delay_storage 

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

        # on repeat_prob % of trials, sample with repetition
        repeat_rv = rng.integers(0,100,1)
        if repeat_rv < 100*self.repeat_prob: 
            letters = rng.choice(self.num_letters, self.list_length, replace=True) 
        else:
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

    def __getitem__(self, idx):

        # every new batch, increment the list_length 
        # once the list length exceeds the max length, return back to 1 
        if idx % self.batch_size == 0: 
            self.list_length += 1
            if self.list_length > self.max_length:
                self.list_length = 1

        X, y = self.construct_trial()

        if self.double_trial: 

            rng = np.random.default_rng()
            uniform_0_1 = rng.random()

            if uniform_0_1 < self.storage_frac:
                # the second list is a fixed delay period followed by recalling the previous list 
                # these are termed storage trials
                list_recall = self.delay_start + self.list_length + self.delay_middle
                X2, y2 = self.recall_list_from_storage(X[list_recall:], y[list_recall:])
            else:
                # the second list is a normal list, i.e. presented letters followed by recalling those letters
                X2, y2 = self.construct_trial()

            X = torch.cat((X, X2),axis=0)
            y = torch.cat((y, y2),axis=0)

        return X.to(torch.float32), y.to(torch.float32) 