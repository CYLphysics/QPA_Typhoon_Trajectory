import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.utils.data as Data


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
        # outputs = Variable(torch.zeros(self.minibatch_size, self.n_step, self.cell_param['output_channels'], input.size()[-2], input.size()[-1]).cuda())
        outputs = []
        # 
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
    def __init__(self, cell_param):
        super(MConvGRUCell, self).__init__(cell_param)
        self.build_model()

    def get_parameter(self,shape,init_method = 'xavier'):
        param = Parameter(torch.Tensor(*shape).cuda())
        if init_method == 'xavier':
            nn.init.xavier_uniform_(param)
        elif init_method == 'zero':
            nn.init.constant_(param,0)
        else:
            raise ('init method error')
        return param

    def build_model(self):

        input_to_state_shape = [
            self.output_dim,
            self.input_dim,
            self.input_to_state_kernel_size[0],
            self.input_to_state_kernel_size[1]
        ]
        state_to_state_shape = [
            self.output_dim,
            self.output_dim,
            self.state_to_state_kernel_size[0],
            self.state_to_state_kernel_size[1]
        ]
        state_bias_shape = [
            1, self.output_dim, 1, 1
        ]

        self.w_xz1 = self.get_parameter(input_to_state_shape)
        self.w_xz2 = self.get_parameter(state_to_state_shape)
        self.w_xz3 = self.get_parameter(state_to_state_shape)
        self.w_hz = self.get_parameter(state_to_state_shape)

        self.w_xr1 = self.get_parameter(input_to_state_shape)
        self.w_xr2 = self.get_parameter(state_to_state_shape)
        self.w_xr3 = self.get_parameter(state_to_state_shape)
        self.w_hr = self.get_parameter(state_to_state_shape)

        self.w_xh1 = self.get_parameter(input_to_state_shape)
        self.w_xh2 = self.get_parameter(state_to_state_shape)
        self.w_xh3 = self.get_parameter(state_to_state_shape)
        self.w_hh = self.get_parameter(state_to_state_shape)

        self.b_z = self.get_parameter(state_bias_shape,'zero')
        self.b_r = self.get_parameter(state_bias_shape,'zero')
        self.b_h_ = self.get_parameter(state_bias_shape,'zero')

    def same_padding(self,kernel_size):
        if kernel_size[0]%2==0 or kernel_size[1]%2==0:
            raise('The kernel size must to be odd if you want padding!')
        else:
            padding = tuple((int((kernel_size[0]-1)/2),int((kernel_size[1]-1)/2)))
        return padding


    def cell(self, x_t, hidden):
        h_tm1 = hidden
        Z = torch.sigmoid(
            F.conv2d(h_tm1, self.w_hz, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            + F.conv2d(
                F.conv2d(
                    F.conv2d(
                        x_t,
                        self.w_xz1,
                        bias=None,
                        padding=self.same_padding(self.input_to_state_kernel_size)
                    ),
                    self.w_xz2,
                    bias = None,
                    padding = self.same_padding(self.state_to_state_kernel_size)
                ),
                self.w_xz3,
                bias = None,
                padding = self.same_padding(self.state_to_state_kernel_size)
            )
            + self.b_z
        )

        R = torch.sigmoid(
            F.conv2d(h_tm1, self.w_hr, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            + F.conv2d(
                F.conv2d(
                    F.conv2d(
                        x_t,
                        self.w_xr1,
                        bias=None,
                        padding=self.same_padding(self.input_to_state_kernel_size)
                    ),
                    self.w_xr2,
                    bias = None,
                    padding = self.same_padding(self.state_to_state_kernel_size)
                ),
                self.w_xr3,
                bias = None,
                padding = self.same_padding(self.state_to_state_kernel_size)
            )
            + self.b_r
        )

        H_ = F.leaky_relu(
            F.conv2d(h_tm1,self.w_hh,bias = None,padding = self.same_padding(self.state_to_state_kernel_size))
            + R*F.conv2d(
                    F.conv2d(
                        F.conv2d(
                            x_t,
                            self.w_xh1,
                            bias=None,
                            padding = self.same_padding(self.input_to_state_kernel_size)
                        ),
                        self.w_xh2,
                        bias = None,
                        padding = self.same_padding(self.state_to_state_kernel_size)
                    ),
                    self.w_xh3,
                    bias = None,
                    padding = self.same_padding(self.state_to_state_kernel_size)
                )
            + self.b_h_,negative_slope = 0.2
        )

        H = (1-Z)*H_ + Z*h_tm1

        return H

    def forward(self, input, hidden):
        h_t = self.cell(input,hidden)
        return h_t
    
    

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
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y


class TrainLoader(Data.Dataset):
    def __init__(self, X_wide_train, X_deep_train, y_train):
        self.X_wide_train = X_wide_train
        self.X_deep_train = X_deep_train
        self.y_train = y_train
        
    def __getitem__(self, index):
        return [self.X_wide_train[index], self.X_deep_train[index]], self.y_train[index]
    
    def __len__(self):
        return len(self.X_wide_train)
    



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.channel_attention = CALayer(4, reduction=2)
        
        self.convGRU1 = MultiConvGRU(cell_param={
            'input_channels' : 4,
            'output_channels' : 64,
            'input_to_state_kernel_size' : (3, 3),
            'state_to_state_kernel_size' : (3, 3)
            }, return_state=False, return_sequence=True)
        
        self.channel_attention_0 = CALayer(64)
        
        self.convGRU2 = MultiConvGRU(cell_param={
            'input_channels' : 64,
            'output_channels' : 128,
            'input_to_state_kernel_size' : (3, 3),
            'state_to_state_kernel_size' : (3, 3)
            }, return_state=False, return_sequence=True)
        
        self.channel_attention_1 = CALayer(128)
        
        self.convGRU3 = MultiConvGRU(cell_param={
            'input_channels' : 128,
            'output_channels' : 256,
            'input_to_state_kernel_size' : (3, 3),
            'state_to_state_kernel_size' : (3, 3)
            }, return_state=False, return_sequence=False)
        
        self.channel_attention_2 = CALayer(256)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.fc1 = nn.Linear(256 * 3 * 3, 128)
        self.fc2 = nn.Linear(53 + 128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, wide, deep):
        
        temp_deep_list = []
        for i in range(deep.shape[1]):
            temp_deep = deep[:, i, :, :, :]
            temp_deep = temp_deep + self.channel_attention(temp_deep)
            temp_deep_list.append(temp_deep)

        deep = torch.stack(temp_deep_list, dim=1)
        
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
    