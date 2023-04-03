from pickle import FALSE
import torch.nn as nn 
import torch

class RNNcell(nn.Module):

    """ Vanilla RNN with:
            - Feedback from output
            - Sigmoid nonlinearity over hidden activations 
            - Softmax activation over output 
            - Initialization follows Botvinick and Plaut, 2006 
            - Incorporated plastic connections based on Miconi, 2018
    """

    def __init__(self, data_size, hidden_size, output_size, noise_std, nonlin,
                bias, feedback_bool, alpha_s, plastic, rule, h2h_weights):

        """ Init model.
        @param data_size (int): Input size
        @param hidden_size (int): the size of hidden states
        @param output_size (int): number of classes
        @param noise_std (float): std. dev. for gaussian noise
        @param nonlin (str): Nonlinearity for hidden activations: sigmoid, relu, tanh, or linear.
        @param h2h_bias (bool): if true, bias units are used for hidden units
        @param feedback_bool (bool): if true, feedback connections are implemented
        @param feedback_size (int): size of feedback units, if None defaults to output_size
        @param plastic (bool): if true, implement hebbian connections 
        """
        super(RNNcell, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nonlin = nonlin
        self.noise_std = noise_std
        self.feedback_bool = feedback_bool
        self.alpha_s = alpha_s
        self.plastic = plastic
        self.rule = rule
        self.h2h_weights = h2h_weights

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

        if self.plastic:
            # plasticity coefficients
            self.alpha =  torch.nn.Parameter(torch.rand((hidden_size, hidden_size))*2 - 1)
            self.alpha.requires_grad = True

            # learning rate for plasticity, we'll allow the network to learn this 
            self.eta = torch.nn.Parameter(torch.Tensor([0.01]))
            self.eta.requires_grad = True

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

    def forward(self, data, h_prev, feedback, hebb, i_prev, device):
        
        """
        @param data: input at time t
        @param r_prev: firing rates at time t-1
        @param x_prev: membrane potential values at time t-1
        @param feedback: feedback from previous timestep
        @param hebb: hebbian weights
        @param i_prev: if using continuous time RNN 
        """
        
        noise = self.noise_std*torch.randn(h_prev.shape).to(device)

        # h2h hebbian connections
        if self.plastic:
            hebb_activity = h_prev@torch.mul(self.alpha, hebb)[0]
        else:
            hebb_activity = torch.zeros((h_prev.shape[0], self.hidden_size)).to(device)

        # Only allow hebbian recurrent weights if false
        if self.h2h_weights:
            h_contribution = self.h2h(h_prev)
        else:
            h_contribution = h_prev

        i = (1-self.alpha_s)*i_prev + self.alpha_s*(hebb_activity + self.i2h(data) + h_contribution 
        + self.o2h(feedback) + noise)
        h = self.F(i)

        if self.plastic:
            if self.rule == 'oja':
                hebb = hebb + self.eta_c * torch.mul((h_prev[0].unsqueeze(1) - 
                torch.mul(hebb , h[0].unsqueeze(0))) , h[0].unsqueeze(0))

            if self.rule == 'decay': 
                hebb = (1-self.eta)*hebb + self.eta*torch.bmm(h_prev.unsqueeze(2), h.unsqueeze(1))[0]

        return h, hebb, i

class RNN_one_layer(nn.Module):

    """ Single layer RNN """

    def __init__(self, input_size, hidden_size, output_size, feedback_bool, bias, 
        nonlin='sigmoid', noise_std=0.0, plastic=False, alpha_s=1.0, rule='decay', 
        h2h_weights=True):

        """ Init model.
        @param data_size: Input size
        @param hidden_size: the size of hidden states
        @param output_size: number of classes
        """
        super(RNN_one_layer, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
            
        self.RNN = RNNcell(input_size, hidden_size, output_size, noise_std, nonlin, 
        bias=bias, feedback_bool=feedback_bool, plastic=plastic, alpha_s=alpha_s,
         rule=rule, h2h_weights=h2h_weights)

        self.h2o = nn.Linear(hidden_size, output_size, bias=bias)
        nn.init.uniform_(self.h2o.weight, -1.0, 1.0)

    def forward(self, data, h_prev, o_prev, hebb_prev, i_prev, device):
        """
        @param data: input at time t
        @param h_prev : firing rates at time t-1 
        @param o_prev: output at time t-1
        """
        h, hebb, i = self.RNN(data, h_prev, o_prev, hebb_prev, i_prev, device)

        output = self.h2o(h)

        return output, h, hebb, i

    def init_states(self, batch_size, device, h0_init_val):

        output = torch.zeros(batch_size, self.output_size).to(device)
        h0 = torch.full((batch_size, self.hidden_size), float(h0_init_val)).to(device)
        hebb0 = torch.zeros(batch_size, self.hidden_size, self.hidden_size).to(device)
        i0 = torch.full((batch_size, self.hidden_size), float(0.0)).to(device)
       
        return output, h0, hebb0, i0