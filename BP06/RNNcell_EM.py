from pickle import FALSE
import torch.nn as nn 
import torch
from EM import EM

# [1] A neural network ...

class RNNcell(nn.Module):

    """ Vanilla RNN with:
            - Feedback from output
            - Sigmoid nonlinearity over hidden activations 
            - Softmax activation over output 
            - Initialization follows Botvinick and Plaut, 2006 
            - Incorporated plastic connections based on Miconi, 2018
    """

    def __init__(self, data_size, hidden_size, output_size, noise_std, nonlin,
                bias, feedback_bool, alpha_s, storage_capacity=2):

        """ Init model.
        :param (int) data_size: Input size
        :param (int) hidden_size: the size of hidden states
        :param (int) output_size : number of classes
        :param (float) noise_std: std. dev. for gaussian noise
        :param (str) nonlin: Nonlinearity for hidden activations: sigmoid, relu, tanh, or linear.
        :param (bool) h2h_bias: if true, bias units are used for hidden units
        :param (bool) feedback_bool: if true, feedback connections are implemented
        """
        super(RNNcell, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nonlin = nonlin
        self.noise_std = noise_std
        self.feedback_bool = feedback_bool
        self.alpha_s = alpha_s

        # recurrent to recurrent connections 
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        nn.init.uniform_(self.h2h.weight, -0.5, 0.5)
        
        # input to recurrent unit connections 
        self.i2h = nn.Linear(data_size, hidden_size, bias=False)
        nn.init.uniform_(self.i2h.weight, -1.0, 1.0)

        # output to recurrent connections 
        # default to output size if no feedback size is specified 
        feedback_size = output_size

        self.o2h = nn.Linear(feedback_size, hidden_size, bias=False)
        nn.init.uniform_(self.o2h.weight, -1.0, 1.0)

        if nonlin == 'sigmoid':
            self.F = nn.Sigmoid()
        if nonlin == 'relu':
            self.F = nn.ReLU()
        if nonlin == 'tanh':
            self.F = nn.Tanh()
        if nonlin == 'linear':
            self.F = nn.Identity()
        if nonlin == 'relu6':
            self.F = nn.ReLU6()

    def forward(self, data, h_prev, feedback, i_prev, device):
        
        """
        @param data: input at time t
        @param r_prev: firing rates at time t-1
        @param x_prev: membrane potential values at time t-1
        @param feedback: feedback from previous timestep
        @param i_prev: if using continuous time RNN 
        """
        
        noise = self.noise_std*torch.randn(h_prev.shape).to(device)

        i = (1-self.alpha_s)*i_prev + self.alpha_s*(self.i2h(data) + self.h2h(h_prev)
        + self.o2h(feedback) + noise)
        h = self.F(i)
    
        return h, i 

class RNN_one_layer_EM(nn.Module):

    """ Single layer RNN """

    def __init__(self, input_size, hidden_size, output_size, feedback_bool, bias, 
        nonlin='sigmoid', noise_std=0.0, alpha_s=1.0, storage_capacity=12, cmpt=0.8):

        """ Init model.
        :param int data_size: Input size
        :param int hidden_size: the size of hidden states
        :param int output_size: number of classes
        :param bool feedback_bool: set to True to allow for feedback projections 
        :param bool bias: Set to True to allow for bias term 
        """
        super(RNN_one_layer_EM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cmpt = cmpt # competition between memories in EM 
        self.F = nn.Sigmoid()
            
        self.RNN = RNNcell(input_size, hidden_size, output_size, noise_std, nonlin, 
        bias=bias, feedback_bool=feedback_bool, alpha_s=alpha_s)

        self.h2o = nn.Linear(hidden_size, output_size, bias=bias)
        nn.init.uniform_(self.h2o.weight, -1.0, 1.0)
        
        self.em = EM(storage_capacity, hidden_size, 'cosine')
        self.hpc = nn.Linear(hidden_size+output_size, 1)

    def forward(self, data, h_prev, o_prev, i_prev, device):
        """
        @param data: input at time t
        @param h_prev : firing rates at time t-1 
        @param o_prev: output at time t-1
        """
        h, i = self.RNN(data, h_prev, o_prev, i_prev, device)

        output_no_EM = self.h2o(h)
        
        hpc_input = torch.cat([h, output_no_EM], dim=1)
        
        EM_gate = self.F(self.hpc(hpc_input))
        
        EM_mem = self.recall_from_EM(h, EM_gate)
        
        EM_mem = EM_mem.to(device)
        
        self.encode_to_EM(h+EM_mem)
        
        h_EM = h + EM_mem
        
        output_EM = self.h2o(h_EM)
        
        return output_EM, h, i
    
    def recall_from_EM(self, c_t, inps_t, comp_t=None):

        """
        Code from [1]
        run the "pattern completion" procedure
        Parameters
        ----------
        c_t : torch.tensor, vector
            cell state
        leak_t : torch.tensor, scalar
            LCA param, leak
        comp_t : torch.tensor, scalar
            LCA param, lateral inhibition
        inps_t : torch.tensor, scalar
            LCA param, input strength / feedforward weights
        Returns
        -------
        tensor, tensor
            updated cell state, recalled item
        """
        if comp_t is None:
            comp_t = self.cmpt

        m_t = self.em.get_memory(c_t, leak=0, comp=comp_t, w_input=inps_t)
            
        return m_t
    
    def encode_to_EM(self, hidden_state):
        
        if self.em.encoding_off == False:
            self.em.save_memory(hidden_state)

    def init_states(self, batch_size, device, h0_init_val):

        output = torch.zeros(batch_size, self.output_size).to(device)
        h0 = torch.full((batch_size, self.hidden_size), float(h0_init_val)).to(device)
        i0 = torch.full((batch_size, self.hidden_size), float(0.0)).to(device)
       
        return output, h0, i0
    
    def encoding_off(self):
        self.em.encoding_off = True
        
    def encoding_on(self):
        self.em.encoding_off = False