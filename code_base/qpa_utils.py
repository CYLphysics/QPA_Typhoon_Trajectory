import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from transformers import get_linear_schedule_with_warmup

import random 
import qadence as qd
import numpy as np 
import time 



chunk_size = 64
qnn_depth = 20

def generate_qubit_states_torch(n_qubit, num_vectors):
    # Calculate the total number of possible combinations
    total_combinations = 2 ** n_qubit
    if num_vectors > total_combinations:
        raise ValueError(f"Number of vectors requested ({num_vectors}) exceeds the total number of unique combinations ({total_combinations}).")
    
    # Generate random unique indices
    random_indices = random.sample(range(total_combinations), num_vectors)
    random_indices = torch.tensor(random_indices, dtype=torch.int64).unsqueeze(1)
    
    # Convert indices to binary representation and then to -1 and 1
    qubit_states = ((random_indices.unsqueeze(2) & (1 << torch.arange(n_qubit, dtype=torch.int64))) > 0).float() * 2 - 1
    
    return qubit_states

class MappingModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        # Initialize layers: an input layer, multiple hidden layers, and an output layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, X):
        X = X.type_as(self.input_layer.weight)
        X = self.input_layer(X)
        for hidden in self.hidden_layers:
            X = hidden(X)
        output = self.output_layer(X)
        return output
    




class QT_HyperNet(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_sub_hypernetwork):
        super(QT_HyperNet, self).__init__()
        
        self.n_sub_hypernetwork = n_sub_hypernetwork
        self.weight_length = int(np.ceil((vocab_size * hidden_size) / self.n_sub_hypernetwork ))

        self.out_dim_MPS = 64
        self.out_dim_MLP = chunk_size
        self.batch_size = int(np.ceil((self.weight_length/self.out_dim_MLP))) #1000 #400 #4000
        self.dropout = nn.Dropout(p=0.)
        # typically self.batch_size * self.out_dim_MLP * n_sub_hypernetwork should > vocab_size * hidden_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_mapping = "MLP"
        self.classical_layers = "MLP"

        
        self.n_qubit_qt = int(np.ceil(np.log2(self.batch_size)))
        self.n_qubit = self.n_qubit_qt
        self.q_depth    = qnn_depth
        
        ansatz = qd.hea(self.n_qubit , self.q_depth )
        ansatz = qd.tag(ansatz, "ansatz")

        circuit = qd.QuantumCircuit(self.n_qubit , ansatz)
        
        self.QuantumNN = qd.QuantumModel(circuit).to(torch.complex128).to(self.device) 

        if self.init_mapping == "MLP":
            self.MappingNetwork = MappingModel(self.n_qubit+1, [32, 32], self.out_dim_MPS)

        if self.classical_layers == "MLP": 
            self.fc1 = nn.Linear(self.out_dim_MPS, self.out_dim_MLP)


    def forward(self):
        
        compute_method = "checkpoint"
                
        state =  self.QuantumNN() # .to(torch.complex64)
        probs_ = torch.abs(state) ** 2
        probs_ = probs_.to(self.device).to(torch.complex128)
        easy_scale_coeff = 2**(self.n_qubit-1)
        gamma = 0.1
        beta  = 0.60
        alpha = 0.15
        # print("qubit number : ", self.n_qubit)
        probs_ = probs_.reshape(2**(self.n_qubit),1)
        probs_ = (beta*torch.tanh(gamma*easy_scale_coeff*probs_))**(alpha) 
        probs_ = probs_ - torch.mean(probs_)
        
        probs_ = probs_.flatten()
        probs_ = probs_[:self.batch_size]
        probs_ = probs_.reshape(self.batch_size, 1, 1)

        
        qubit_states_torch = generate_qubit_states_torch(self.n_qubit, self.batch_size)[:self.weight_length].to(self.device)
        combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=2)
        if self.init_mapping == "MPS":
            combined_data_torch = combined_data_torch.reshape(self.batch_size,  self.n_qubit + 1)


        prob_val_post_processed_list = []
        if compute_method == "checkpoint":          
            
            batch_data = combined_data_torch[0:self.batch_size]
            batch_data.requires_grad_()
            
            prob_val_post_processed_batch = checkpoint(self.MappingNetwork, batch_data)

            if self.classical_layers == "MLP":
                
                prob_val_post_processed_batch = checkpoint(self.dropout, prob_val_post_processed_batch)
                prob_val_post_processed_batch = checkpoint(self.fc1, prob_val_post_processed_batch)
                              
            
            prob_val_post_processed_list.append(prob_val_post_processed_batch)
                      
            torch.cuda.empty_cache()

        prob_val_post_processed_list = prob_val_post_processed_list[:self.weight_length]
        prob_val_post_processed = torch.cat(prob_val_post_processed_list, dim=0)
        
        prob_val_post_processed = prob_val_post_processed.view(-1)[:self.weight_length]
        prob_val_post_processed = prob_val_post_processed - prob_val_post_processed.mean()
        
        torch.cuda.empty_cache()

        return prob_val_post_processed
    
    
class QTLoRALayer(nn.Module):

    def __init__(self, original_layer, r, alpha, n_sub_hypernetwork=1):
        super(QTLoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(p=0.0)
        self.dtype = torch.float32  # Ensure all tensors are of this type

        # Generate the parameters of A and B
        self.grand_hypernetwork = nn.ModuleList([
            QT_HyperNet(
                original_layer.weight.size(0)*r + r*original_layer.weight.size(1),  
                1, 
                n_sub_hypernetwork)
            for _ in range(n_sub_hypernetwork)]).cuda()
        

        # Freeze the original layer's parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):

        gen_weights = []
        for sub_hypernetwork in self.grand_hypernetwork:
            gen_weights.append(sub_hypernetwork())
        self.generated_weights = torch.cat(gen_weights, dim=0).view(-1)[:self.original_layer.weight.size(0)*self.r + self.r*self.original_layer.weight.size(1)].cuda()
        self.generated_weights_A = self.generated_weights[:self.original_layer.weight.size(0)*self.r].view(self.original_layer.weight.size(0), self.r).type(self.dtype)
        self.generated_weights_B = self.generated_weights[self.original_layer.weight.size(0)*self.r:].view(self.r, self.original_layer.weight.size(1)).type(self.dtype)

        batch_size, hidden_size = x.size()
        x_reshaped = self.dropout(x).view(-1, hidden_size).type(self.dtype)
        delta = (x_reshaped @ self.generated_weights_B.t()) @ self.generated_weights_A.t()
        delta = delta * (self.alpha / self.r)
        delta = delta.view(batch_size, self.generated_weights_A.shape[0])
        x = x.to(torch.float32)
        self.original_layer = self.original_layer.to(torch.float32)
        
        
        return self.original_layer(x) + delta   
    
class QTLoRAConv2d(nn.Module):
    def __init__(self, original_conv, r=4, alpha=8, n_sub_hypernetwork = 1):
        super(QTLoRAConv2d, self).__init__()
        self.original_conv = original_conv
        self.r = r
        self.alpha = alpha
        self.dtype = torch.float32  # Ensure all tensors are of this type


        # Generate the parameters of A and B
        self.grand_hypernetwork = nn.ModuleList([
            QT_HyperNet(
                original_conv.out_channels*r + r*original_conv.in_channels*(original_conv.kernel_size[0]*original_conv.kernel_size[1]),  
                1, 
                n_sub_hypernetwork)
            for _ in range(n_sub_hypernetwork)]).cuda()
        
        
        # Freeze the original convolutional layer
        for param in self.original_conv.parameters():
            param.requires_grad = False

    def forward(self, x):

        gen_weights = []
        for sub_hypernetwork in self.grand_hypernetwork:
            gen_weights.append(sub_hypernetwork())
        self.generated_weights = torch.cat(gen_weights, dim=0).view(-1)[:self.original_conv.out_channels*self.r + self.r*self.original_conv.in_channels*(self.original_conv.kernel_size[0]*self.original_conv.kernel_size[1])].cuda()
        self.generated_weights_A = self.generated_weights[:self.original_conv.out_channels*self.r].view(self.original_conv.out_channels, self.r, 1, 1).type(self.dtype)
        self.generated_weights_B = self.generated_weights[self.original_conv.out_channels*self.r:].view(self.r, self.original_conv.in_channels, *self.original_conv.kernel_size).type(self.dtype)

        # Standard convolution output
        x =  x.to(torch.float32).cuda()
        conv_out = self.original_conv(x)

        # Compute low-rank adaptation
        lora_out = F.conv2d(x, self.generated_weights_B, bias=None, padding=self.original_conv.padding)
        lora_out = F.conv2d(lora_out, self.generated_weights_A, bias=None, padding=0)

        # Scale LoRA output
        lora_out = lora_out * (self.alpha / self.r)

        return conv_out + lora_out


class LoRALayer(nn.Module):
    def __init__(self, original_layer, r, alpha):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(p=0.0)  # No dropout by default
        self.dtype = torch.float32  # Ensure all tensors are of this type

        # LoRA weight matrices
        self.A = nn.Parameter(torch.randn(original_layer.weight.size(0), r) * 0.01).type(self.dtype).cuda()
        self.B = nn.Parameter(torch.randn(r, original_layer.weight.size(1)) * 0.01).type(self.dtype).cuda()

        # Freeze the original layer's parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size, hidden_size = x.size()
        x_reshaped = self.dropout(x).view(-1, hidden_size).type(self.dtype)
        delta = (x_reshaped @ self.B.t()) @ self.A.t()
        delta = delta * (self.alpha / self.r)
        delta = delta.view(batch_size, self.A.shape[0])
        x = x.to(torch.float32)
        

        return self.original_layer(x) + delta  
    
class LoRAConv2d(nn.Module):
    def __init__(self, original_conv, r=4, alpha=8):
        super(LoRAConv2d, self).__init__()
        self.original_conv = original_conv
        self.r = r
        self.alpha = alpha

        # Create LoRA matrices
        self.A = nn.Parameter(torch.randn(original_conv.out_channels, r, 1, 1) * 0.01)
        self.B = nn.Parameter(torch.randn(r, original_conv.in_channels, *original_conv.kernel_size) * 0.01)

        # Freeze the original convolutional layer
        for param in self.original_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Standard convolution output
        x =  x.to(torch.float32).cuda()
        conv_out = self.original_conv(x)

        # Compute low-rank adaptation
        lora_out = F.conv2d(x, self.B, bias=None, padding=self.original_conv.padding)
        lora_out = F.conv2d(lora_out, self.A, bias=None, padding=0)

        # Scale LoRA output
        lora_out = lora_out * (self.alpha / self.r)

        return conv_out + lora_out
    
class ConvRNN(nn.Module):
    def __init__(self,
                 cell_param,
                 return_state,
                 return_sequence,
                 ):
        super(ConvRNN, self).__init__()
        self.cell_param = cell_param
        self.return_state =return_state
        self.return_sequence = return_sequence

    def init_paramter(self,shape):
        return Variable(torch.zeros(shape).cuda())

    def forward(self, input, state=None):

        if state is None:
            state = self.init_hidden(input)
        else:
            state = state
        # batch_size
        self.minibatch_size = input.size()[0]
        # time_step
        self.n_step = input.size()[1]
        outputs = []

        for i in range(self.n_step):
            x_t = input[:, i, :, :, :]
            state = self.cell(x_t, state)
            if type(state) == type((1,2)): 
                outputs.append(state[0]) 
            else:
                outputs.append(state)
        outputs = torch.stack(outputs,dim=1)
        self.outputs = outputs
        if self.return_sequence:
            if self.return_state:
                return outputs, state
            else:
                return outputs
        else:
            if self.return_state:
                return state
            else:
                if type(state) == type((1)): # int
                    return state[0]
                else:
                    return state


class ConvRNNCell(nn.Module):
    def __init__(self, cell_param):
        super(ConvRNNCell, self).__init__()
        self.input_dim = cell_param['input_channels']
        self.output_dim = cell_param['output_channels']
        self.input_to_state_kernel_size = cell_param['input_to_state_kernel_size']
        self.state_to_state_kernel_size = cell_param['state_to_state_kernel_size']



class MultiConvGRU(ConvRNN):
    def __init__(self,
                 cell_param,
                 return_state,
                 return_sequence):
        super(MultiConvGRU, self).__init__(
            cell_param,
            return_state,
            return_sequence
        )

        self.build()

    def build(self):
        self.cell = MConvGRUCell(self.cell_param)

    def init_hidden(self, input):

        hidden_size = (input.size()[0], self.cell_param['output_channels'], input.size()[-2], input.size()[-1])
        h = Variable(self.init_paramter(hidden_size))
        state = h

        return state



class MConvGRUCell(ConvRNNCell):
    def __init__(self, cell_param, r=4, alpha=8):
        super(MConvGRUCell, self).__init__(cell_param)

        self.r = r
        self.alpha = alpha

        # Convert standard convolutions to LoRA-enhanced convolutions
        self.w_xz1 = LoRAConv2d(nn.Conv2d(self.input_dim, self.output_dim, kernel_size=self.input_to_state_kernel_size, padding='same'), r, alpha)
        self.w_xz2 = LoRAConv2d(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=self.state_to_state_kernel_size, padding='same'), r, alpha)
        self.w_xz3 = LoRAConv2d(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=self.state_to_state_kernel_size, padding='same'), r, alpha)
        self.w_hz = LoRAConv2d(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=self.state_to_state_kernel_size, padding='same'), r, alpha)

        self.w_xr1 = LoRAConv2d(nn.Conv2d(self.input_dim, self.output_dim, kernel_size=self.input_to_state_kernel_size, padding='same'), r, alpha)
        self.w_xr2 = LoRAConv2d(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=self.state_to_state_kernel_size, padding='same'), r, alpha)
        self.w_xr3 = LoRAConv2d(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=self.state_to_state_kernel_size, padding='same'), r, alpha)
        self.w_hr = LoRAConv2d(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=self.state_to_state_kernel_size, padding='same'), r, alpha)

        self.w_xh1 = LoRAConv2d(nn.Conv2d(self.input_dim, self.output_dim, kernel_size=self.input_to_state_kernel_size, padding='same'), r, alpha)
        self.w_xh2 = LoRAConv2d(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=self.state_to_state_kernel_size, padding='same'), r, alpha)
        self.w_xh3 = LoRAConv2d(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=self.state_to_state_kernel_size, padding='same'), r, alpha)
        self.w_hh = LoRAConv2d(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=self.state_to_state_kernel_size, padding='same'), r, alpha)

        # Bias terms remain unchanged
        self.b_z = nn.Parameter(torch.zeros(1, self.output_dim, 1, 1))
        self.b_r = nn.Parameter(torch.zeros(1, self.output_dim, 1, 1))
        self.b_h_ = nn.Parameter(torch.zeros(1, self.output_dim, 1, 1))

    def cell(self, x_t, hidden):
        h_tm1 = hidden
        Z = torch.sigmoid(
            self.w_hz(h_tm1) + self.w_xz3(self.w_xz2(self.w_xz1(x_t))) + self.b_z
        )
        R = torch.sigmoid(
            self.w_hr(h_tm1) + self.w_xr3(self.w_xr2(self.w_xr1(x_t))) + self.b_r
        )
        H_ = F.leaky_relu(
            self.w_hh(h_tm1) + R * self.w_xh3(self.w_xh2(self.w_xh1(x_t))) + self.b_h_,
            negative_slope=0.2
        )
        H = (1 - Z) * H_ + Z * h_tm1

        return H

    def forward(self, input, hidden):
        return self.cell(input, hidden)



## Channel Attention (CA) Layer
class CALayer(nn.Sequential):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.conv_du = nn.Sequential(
                # global average pooling: feature --> point
                nn.AdaptiveAvgPool2d(1),
                # feature channel downscale and upscale --> channel weight
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        ).to(torch.float32)

    def forward(self, x):
        x = x.to(torch.float32).cuda()
        y = self.conv_du(x)
        return x * y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.channel_attention = CALayer(4, reduction=2)

        self.convGRU1 = MultiConvGRU(cell_param={
            'input_channels': 4,
            'output_channels': 64,
            'input_to_state_kernel_size': (3, 3),
            'state_to_state_kernel_size': (3, 3)
        }, return_state=False, return_sequence=True).to(torch.float32)

        self.channel_attention_0 = CALayer(64)

        self.convGRU2 = MultiConvGRU(cell_param={
            'input_channels': 64,
            'output_channels': 128,
            'input_to_state_kernel_size': (3, 3),
            'state_to_state_kernel_size': (3, 3)
        }, return_state=False, return_sequence=True).to(torch.float32)

        self.channel_attention_1 = CALayer(128)

        self.convGRU3 = MultiConvGRU(cell_param={
            'input_channels': 128,
            'output_channels': 256,
            'input_to_state_kernel_size': (3, 3),
            'state_to_state_kernel_size': (3, 3)
        }, return_state=False, return_sequence=False).to(torch.float32)

        self.channel_attention_2 = CALayer(256)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Apply LoRA to Fully Connected Layers
        self.fc1 = QTLoRALayer(nn.Linear(256 * 3 * 3, 256), r=4, alpha=8).to(torch.complex128)
        self.fc2 = QTLoRALayer(nn.Linear(53 + 256, 128), r=4, alpha=8).to(torch.complex128)
        self.fc3 = LoRALayer(nn.Linear(128, 2), r=4, alpha=8).to(torch.float32)

    def forward(self, wide, deep):
        temp_deep_list = []
        for i in range(deep.shape[1]):
            temp_deep = deep[:, i, :, :, :]
            temp_deep = temp_deep + self.channel_attention(temp_deep)
            temp_deep_list.append(temp_deep)

        deep = torch.stack(temp_deep_list, dim=1)
        deep = deep.to(torch.complex128).cuda()
        deep = self.convGRU1(deep)
        

        temp_deep_list = []
        for i in range(deep.shape[1]):
            temp_deep = deep[:, i, :, :, :]
            temp_deep = temp_deep + self.channel_attention_0(temp_deep)
            temp_deep = self.pool(temp_deep)
            temp_deep_list.append(temp_deep)

        deep = torch.stack(temp_deep_list, dim=1)
        
        deep = self.convGRU2(deep)
        
        temp_deep_list = []
        for i in range(deep.shape[1]):
            temp_deep = deep[:, i, :, :, :]
            temp_deep = temp_deep + self.channel_attention_1(temp_deep)
            temp_deep = self.pool(temp_deep)
            temp_deep_list.append(temp_deep)

        deep = torch.stack(temp_deep_list, dim=1)
        
        _h = self.convGRU3(deep)
        
        _h = _h + self.channel_attention_2(_h)
        
        deep = self.pool(_h)
        
        deep = deep.view(-1, 256 * 3 * 3)
        
        deep = self.fc1(deep)
        wide = wide.view(-1, 53)
        wide_n_deep = torch.cat((wide, deep),1)
        
        wide_n_deep = F.relu(self.fc2(wide_n_deep))
        wide_n_deep = F.relu(self.fc3(wide_n_deep))
        
        
        return wide_n_deep