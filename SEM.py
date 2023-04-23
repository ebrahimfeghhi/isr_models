import numpy as np
import matplotlib.pyplot as plt 
from analyze_errors import analyze_errors

    
class SEM(analyze_errors):

    '''
    This class implements to multi-trial version of SEM, as described by Henson (1998).
    Limited to grouping at a single level, will extend later if needed.
    '''

    def __init__(self, param_dict):

        '''
         param_dict contains the following keys: 
        :param float s0: initial strength of start marker 
        :param float e0: initial strength of end marker
        :param float S: change rate for start marker
        :param float E: change rate for end marker
        :param float E_c: contextual drift
        :param float G_c: std. dev. of zero mean gaussian dist. for recall selection process 
        :param float G_p: std. dev. of zero mean gaussian dist. for phonological selection process 
        :param float A_p: phonological representation
        :param flaot P_s: phonological similarity for confusable items
        :param float P_d: phonological similarity for non-confusable items 
        :param float R_s: decay rate for response suppression 
        :param float R_p: decay rate for phonological activations
        :param float T_o: threshold for response 
        :param int C_p: number of episodes between presentation of each item 
        :param int C_d: number of episodes between during the retention interval 
        :param int C_r: number of episodes between recall of each item 
        :param int C_i: number of episodes between trials
        :param int C_a: additional contextual change between trials 
        :param int vocab_size_SM: number of items/items 
        :param bool single_trial_bool: If true, run SEM in single trial mode. 
        :param str saveFolder: Folder to save figures in.  
        '''
        
        self.list_length = param_dict['list_length']
        self.group_size = self.list_length
        self.s0 = param_dict['s0'] 
        self.e0 = param_dict['e0'] 
        self.S = param_dict['S'] 
        self.E = param_dict['E'] 
        self.E_c = param_dict['E_c'] 
        self.E_l = param_dict['E_l'] 
        self.G_c = param_dict['G_c'] 
        self.G_p = param_dict['G_p']
        self.A_p = param_dict['A_p']
        self.P_s = param_dict['P_s']
        self.P_d = param_dict['P_d']
        self.R_s = param_dict['R_s']
        self.R_p = param_dict['R_p']
        self.T_o = param_dict['T_o']
        self.C_p = param_dict['C_p']
        self.C_d = param_dict['C_d']
        self.C_r = param_dict['C_r']
        self.C_i = param_dict['C_i']
        self.C_a = param_dict['C_a']
        self.omission = param_dict['omission_val']
        self.vocab_size_SM = param_dict['vocab_size_SM'] # number of types in LTM 
        self.max_tokens = param_dict['max_tokens'] # max number of tokens that can be stored, somewhat arbitrary
        self.single_trial_bool = param_dict['single_trial']
        self.save_folder = param_dict['saveFolder']
        self.present_context = 1
        self.suppression = 1 
        self.list_position_tokens = [] # stores start and end markers for list position
        self.group_position_tokens = [] # stores start and end markers for group position 
        self.stored_tokens = [] # stores only tokens that are in SEM, this is used by SEM
        self.context_tokens = [] # stores context markers 
        self.response_suppression = np.zeros(self.vocab_size_SM)
        self.phonological_activations = np.zeros(self.vocab_size_SM)
        self.num_correct = 0 # number of lists recalled correctly
        self.num_omissions = 0 
        self.num_intrusions = 0
        self.num_immediate_intrusions = 0 
        self.num_output_protrusions = 0 
        self.recalled_list = [] # stores recalled items of current trials 
        self.current_list = [] # stores items of currently presented list
        self.all_stored_tokens = [] 
        self.all_context_tokens = [] 
        self.all_list_position_tokens = []
        self.all_group_position_tokens = []
        self.num_reps = 0 
        
        super().__init__() # init analyze errors class 

    def start_end_markers(self, list_pos, group_pos):

        '''
        Generates start and end markers indicating list position and group position. 

        :param int list_length: length of entire list
        :param int group_size: size of a group 
        :param int list_pos: position of item on list
        :param int group_num: group number that item belongs in 
        '''
        list_pos_marker = np.asarray([np.round(self.s0*self.S**(list_pos-1.0),6), np.round(self.e0*self.E**(self.list_length-list_pos), 6)])
        group_pos_marker = np.asarray([np.round(self.s0*self.S**(group_pos-1.0),6), np.round(self.e0*self.E**(self.group_size-group_pos), 6)])

        return list_pos_marker, group_pos_marker

    def overlap_function(self, p_j, p_t):

        '''
        Computes the overlap between between two positional/contextual vectors 

        :param list p_j: 2x1  positional for response j, or 1x1  contextual marker 
        :param list p_t: 2x1 positional cue for item presented 
        at time t during the presentation phase
        '''

        exp_term = np.exp(-np.sum((p_t - p_j)**2)**.5)

        return np.round(np.dot(p_j, p_t)**.5 * exp_term, 5)

    def positional_cues(self, plot=False):


        s_arr = [] # start marker array
        e_arr = [] # end marker array

        for i in range(self.list_length):
            lp, _ = self.start_end_markers(i+1, i+1)
            s_arr.append(lp[0])
            e_arr.append(lp[1])            
        if plot: 
            plt.plot(s_arr, marker='o')
            plt.plot(e_arr, marker='o')
            plt.title("Positional cues")
            plt.show()
        return s_arr, e_arr

    def positional_overlap(self, plot=False):
        plt.figure(figsize=(8,6))
        for i in range(self.list_length):
            pos_sim = []
            for j in range(self.list_length):
                se, _ = self.start_end_markers(i+1, i+1)
                se2, _ = self.start_end_markers(j+1, j+1)
                o = self.overlap_function(se, se2)
                pos_sim.append(o)
            if plot:
                plt.plot(pos_sim, marker='o', label=f'Pos {i+1}')
        if plot:
            plt.ylim(0,1.2)
            plt.title("Positional overlap")
            plt.legend()
            plt.show()

        return pos_sim 

    def add_token(self, item, list_position, group_position, phono_activation, context, suppression):

        '''
        Add tokens and modify phonological + suppression type representations 

        :param int item: integer corresponding to item 
        :param list list_position: start and end values coding list position
        :param list group_position: start and end values coding group position
        :param float phono_activation: how much to boost phonological type representation 
        :param float context: context value of token 
        :param float suppression: response suppression value of type representation 
        '''
        
        # ensure stored tokens are all the same length 
        assert len(self.stored_tokens) == len(self.list_position_tokens) == \
        len(self.group_position_tokens) == len(self.context_tokens), \
        print("Token lengths are not matching.")

        # if at max limit, pop first token 
        if len(self.stored_tokens) == self.max_tokens:
            self.stored_tokens.pop(0)
            self.list_position_tokens.pop(0)
            self.group_position_tokens.pop(0)
            self.context_tokens.pop(0)

        self.stored_tokens.append(item)
        self.list_position_tokens.append(list_position)
        self.group_position_tokens.append(group_position)
        self.context_tokens.append(context)

        # modify response suppression and phonological attributes of type representations
        self.response_suppression[item] = suppression
        self.phonological_activations[item] = phono_activation

    def list_presentation(self, list_length, group_size, items):

        '''
        :param int list_length: 
        :param int group_size 
        :param list items: each element is an integer corresponding to a item(0 is A, 25 is Z)
        '''

        # before presenting new items
        
        self.current_list = items 
        self.previously_recalled_list = self.recalled_list
        self.recalled_list = []
        self.list_length = list_length
        self.group_size = group_size
        
        # reset response suppression to 0 for single trial SEM 
        if self.single_trial_bool:
            self.response_suppression *= 0 
        else:
            self.contextual_change_acrosstrials()
        
        for lp, item in enumerate(items):
            
            # between presentation of each item incorporate C_p episodes of 
            # contextual + phonological + response suppression (rs) decay
            if lp != 0:
                for i in range(self.C_p):
                    self.decay_context_phono_rs()

            # obtain start and end markers for list and group position 
            gp = lp % group_size
            lp_marker, gp_marker = self.start_end_markers(lp+1, gp+1)
            self.add_token(item, lp_marker, gp_marker, self.A_p, self.present_context, 1-self.suppression)

        # model effects of retention interval
        for i in range(self.C_d):
            self.decay_context_phono_rs()

    def contextual_change_acrosstrials(self):
        
        # contextual decay for C_a + C_i episodes
        for i in range(self.C_a+self.C_i):
            self.context_tokens = [x*self.E_c for x in self.context_tokens]
            self.list_position_tokens = [x*self.E_l for x in self.list_position_tokens]

            # phonological decay for C_i episodes 
            if i > self.C_a:
                self.phonological_activations *= np.exp(-self.R_p)

    def decay_context_phono_rs(self):

        '''
        Performs one timestep of decay on contextual + phonological informatiton 
        '''

        self.context_tokens = [x*self.E_c for x in self.context_tokens]
        self.phonological_activations *= np.exp(-self.R_p)
        self.response_suppression *= np.exp(-self.R_s)


    def recall_selection(self, recall_list_cue, recall_group_cue, grouping=False, recalled_item=None):

        '''
        Input: 
        :param int recall_list_cue: list position used to initiate recall 
        :param int recall_group_cue: group position used to initiate recall 
        :param bool grouping: if False, ignore effects of grouping
        :param int recalled_item: pass in item from primacy model if recalled, otherwise default is None 

        Output: 
        item selected for retrieval, or if no item was selected an omission response. 
        '''

        # obtain start and end markers based on cue
        rl_query, rg_query = self.start_end_markers(recall_list_cue, recall_group_cue)

        if recalled_item == None:

            # init. all type strengths to 0 
            # although SEM stores episodic tokens in STM, the response competition phase
            # is performed over the type representations in LTM.
            type_strengths = np.zeros(self.vocab_size_SM)

            # compute overlap between retrieval position + contextual cues and stored tokens
            # store the maximum overlap strength for each type representation 
            for i in range(len(self.list_position_tokens)):  
                
                # record item 
                item = int(self.stored_tokens[i])

                # skip if item is an omission response 
                if item == self.omission:
                    continue

                # overlap between query list position (lp) and stored list position tokens 
                o_lp = self.overlap_function(rl_query, self.list_position_tokens[i])
                
                if grouping:
                    o_gp = self.overlap_function(rg_query, self.group_position_tokens[i])
                else:
                    o_gp = 1 # if no grouping, set group overlap strength to 1 

                o_c = self.overlap_function(self.present_context, self.context_tokens[i]) # will bias retrieval towards most recent tokens 

                overlap_strength = o_lp*o_gp*o_c*(1-self.response_suppression[item]) 
                
                # only store max overlap strength value for each item 
                if type_strengths[item] < overlap_strength:
                    type_strengths[item] = overlap_strength
                    
            # add noise to type strengths 
            type_strengths += np.random.default_rng().normal(0, self.G_c, type_strengths.shape[0])

            recalled_item = self.phonological_selection(type_strengths)
              
        correct_item = self.current_list[recall_list_cue-1]
            
        self.compute_errors(correct_item, recalled_item, recall_position=recall_list_cue-1)

        self.recalled_list.append(recalled_item)
        
        # decay values after recalling an item 
        for i in range(self.C_r):
            self.decay_context_phono_rs()

        # recode recalled_item as a new token if operating in multi-trial mode
        if self.single_trial_bool:
            self.response_suppression  *= np.exp(-self.R_s)
            if recalled_item != self.omission:
                self.response_suppression[recalled_item] = self.suppression # suppress item that was just output 
        else:
            # if omission response, record everything as -1 
            if recalled_item == self.omission:
                self.add_token(self.omission, np.asarray([self.omission,self.omission]), 
                            np.asarray([self.omission,self.omission]), self.omission, self.omission, self.omission)
            else:
                self.add_token(int(recalled_item), rl_query, rg_query, self.A_p, self.present_context, self.suppression)

        return recalled_item 

    def phonological_sim(self, item1, item2):

        if item1 == item2:
            return 1
            
        # assume all items are non-confusable for now 
        else:
            return self.P_d

    def phonological_selection(self, type_strengths):

        # strongest item from cueing stage 
        strongest_item = np.argmax(type_strengths)
        type_strengths_phono = np.zeros(self.vocab_size_SM)
        
        # incorporate effects of phonological similarity
        for item, ts in enumerate(type_strengths):

            # provide a boost to items that are phonologically similar to the strongest activated item 
            phono_boost = self.phonological_sim(item, strongest_item) * \
            self.phonological_activations[strongest_item]*(1-self.response_suppression[item])
            type_strengths_phono[item] = ts + phono_boost

        type_strengths_phono += np.random.default_rng().normal(0, self.G_p, type_strengths_phono.shape[0])
        
        max_activation = np.max(type_strengths_phono)

        if max_activation > self.T_o: 
            recalled_item = np.argmax(type_strengths_phono)
        else:
            recalled_item = self.omission

        return recalled_item

    def simulate_trials_SEM(self, num_trials):
        
        self.num_trials = num_trials
        
        self.presented_list_storage = np.zeros((num_trials, self.list_length))
        self.recalled_list_storage = np.zeros((num_trials, self.list_length))

        for trial in range(num_trials):

            if self.single_trial_bool:
                vocab = np.arange(0,self.vocab_size_SM,1)
                
            # for multi-trial SEM, alternate vocab so consecutive lists have non-overlapping items
            else: 
                if trial % 2 == 0:
                    vocab = np.arange(0,self.vocab_size_SM,2)
                else:
                    vocab = np.arange(1,self.vocab_size_SM,2)

            # sample list length items from vocab size 
            current_list= np.random.default_rng().choice(vocab, self.list_length, replace=False)
            self.list_presentation(self.list_length, self.group_size, current_list)

            # recall items
            for i in range(1, self.list_length+1, 1):
                _ = self.recall_selection(i,i)
                
            # u is unique items
            # c is number of times they appear in the recalled list
            _, c = np.unique(self.recalled_list, return_counts=True)
            self.num_reps += np.argwhere(c>1).shape[0]
            
            self.compute_acc()
            
            self.compute_input_omissions()

            self.presented_list_storage[trial] = self.current_list
            self.recalled_list_storage[trial] = self.recalled_list
            
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
            item_act_noisy = self.item_activations + np.random.default_rng().normal(0, self.N, self.vocab_size_PM)

            # retrieve strongest activated item 
            selected_item = np.argmax(item_act_noisy)
            selected_item_act = np.max(item_act_noisy)       

            # add noise to selected item before comparing to output threshold 
            selected_item_act += np.random.default_rng().normal(0, self.M, 1)[0]
            
            # check if activation of selected item is greater than the threshold
            if selected_item_act >= self.T:
                recalled_item = selected_item
                # set activation of selected item to 0 to model response suppression 
                self.item_activations[selected_item] = 0 
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