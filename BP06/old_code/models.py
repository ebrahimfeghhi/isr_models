class encoder_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, feedback_bool, nonlin='sigmoid', 
                noise_std=0.0, recall_cue=10):

        """ Init model.
        @param data_size: Input size
        @param hidden_size: the size of hidden states
        @param output_size: number of classes
        @param recall cue: digit corresponding to recall cue 
        """
        super(encoder_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.recall_cue = recall_cue
            
        self.encoder = RNNcell(input_size, hidden_size, output_size, noise_std, nonlin, feedback_bool=feedback_bool)
        self.decoder = RNNcell(input_size, hidden_size, output_size, noise_std, nonlin, feedback_bool=feedback_bool)

        self.h2o_enc = nn.Linear(hidden_size, output_size, bias=False)
        nn.init.uniform_(self.h2o_enc.weight, -1.0, 1.0)

        self.h2o_dec = nn.Linear(hidden_size, output_size, bias=False)
        nn.init.uniform_(self.h2o_dec.weight, -1.0, 1.0)

    def forward(self, data, h_prev, o_prev, device):

        """
        @param data: input at time t
        @param h_prev : firing rates at time t-1 
        @param o_prev: output at time t-1
        """

        # if recall cue is not present, use encoder RNN
        if torch.argmax(data, dim=1)[0] != self.recall_cue: 
            h = self.encoder(data, h_prev, o_prev, device)
            output = self.h2o_enc(h)
 
        # use decoder RNN for recall stage
        else:
            h = self.decoder(data, h_prev, o_prev, device)
            output = self.h2o_dec(h)

        return output, h

    def init_states(self, batch_size, device, h0_init_val):

        h0 = torch.full((batch_size, self.hidden_size), float(h0_init_val)).to(device)
        output = torch.zeros(batch_size, self.output_size).to(device)

        return output, h0    

class RNN_two_layers(nn.Module):

    """ Multilayer RNN """

    def __init__(self, input_size, hidden_size, output_size, feedback_scaling, nonlin, fb_type):

        """ Init model.
        @param data_size: Input size
        @param hidden_size (list): the size of hidden states
        @param output_size: number of classes
        @param feedback_scaling: scaling factor applied to o2h weights
        @param loss_fn: kl or ce
        @param nonlin: sigmoid, relu, tanh, or linear
        @param fb_type: feedback_type, 0 for feedback from h2 and output to h1, and 1 for feedback only from h2 to h1.
        h2 receives feedback from output in both cases. 
        """
        super(RNN_two_layers, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fb_type = fb_type
        
        self.num_layers = len(hidden_size)

        if fb_type == 0:
            self.RNN1 = RNNcell(input_size, hidden_size[0], hidden_size[1]+output_size, feedback_scaling[0], nonlin)
            self.RNN2 = RNNcell(hidden_size[0], hidden_size[1], output_size, feedback_scaling[1], nonlin)
        if fb_type == 1:
            self.RNN1 = RNNcell(input_size, hidden_size[0], hidden_size[1], feedback_scaling[0], nonlin)
            self.RNN2 = RNNcell(hidden_size[0], hidden_size[1], output_size, feedback_scaling[1], nonlin)

        self.h2o = nn.Linear(hidden_size[1], output_size)
        nn.init.uniform_(self.h2o.weight, -1.0, 1.0)
        nn.init.uniform_(self.h2o.bias, -1.0, 1.0)

    def forward(self, data, h_prev, o_prev):
        """
        @param data: input at time t for first layer
        @param h_prev (list): firing rates at time t-1 for both layers 
        @param o_prev tensor: output at time t-1 for last layer
        """
        if self.fb_type == 0:
            h1 = self.RNN1(data, h_prev[0], torch.cat((h_prev[1], o_prev),dim=1))
        elif self.fb_type == 1:
            h1 = self.RNN1(data, h_prev[0], h_prev[1])

        h2 = self.RNN2(h1, h_prev[1], o_prev)
        output = self.h2o(h2)

        return output, [h1, h2]

    def init_states(self, batch_size, device):

        h0_1 = torch.full((batch_size, self.hidden_size[0]), .5).to(device)
        h0_2 = torch.full((batch_size, self.hidden_size[1]), .5).to(device)
        output = torch.zeros(batch_size, self.output_size).to(device)

        return output, [h0_1, h0_2]