
import torch
import torch.nn as nn
import torch.nn.functional as F

class Aligner(object):
    def __init__(self, args):
        super(Aligner, self).__init__()
        self.args = args
        self.model_dims = {
            'llama3.1': 4096,
            'mistral': 4096,
            'mistral0.1': 4096,
            'bloomz': 4096,
            'gptj': 4096,
            'qwen': 4096,
            'gemma2': 3584
        }

    def setup_aligner(self, model_name):
        # assert self.args.aligner_rank < self.args.category_sample_num, "kernel rank should be smaller than data size"
        src_model_dim = self.model_dims[self.args.model]
        tgt_model_dim = self.model_dims[model_name]
        lyr_num = self.args.aligner_layer_num
        aligner = MLP(self.args, src_model_dim, [src_model_dim] * (lyr_num-1), tgt_model_dim).cuda()
        setattr(self, model_name.replace('.','_'), aligner)

    def get_aligner(self, model_name):
        return getattr(self, model_name.replace('.','_'))

    def forward_aligner(self, model_name, x):
        return getattr(self, model_name.replace('.','_')).forward(x)

class OutputHeadAligner(Aligner):
    def setup_aligner(self, model_name):        
        src_model_dim = self.model_dims[self.args.model]
        tgt_model_dim = self.model_dims[model_name]
        
        aligner = nn.Linear(src_model_dim, tgt_model_dim, bias=False).cuda()

        # the weights are set when we actually load the model.
        
        setattr(self, model_name.replace('.','_'), aligner)

class MLP(nn.Module):
    def __init__(self, args, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        self.args = args

        layers = []
        
        if len(hidden_sizes) > 0 :
            layers.append(nn.Linear(input_size, hidden_sizes[0], bias=False))

            for i in range(1, len(hidden_sizes)):
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], bias=False))

            layers.append(nn.Linear(hidden_sizes[-1], output_size, bias=False))

        else :
            layers.append(nn.Linear(input_size, output_size, bias=False))

        self.layers = nn.ModuleList(layers)
    

    def forward(self, x):
        if len(self.layers) == 1 :
            return self.layers[0](x)

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)  
         
        x = self.layers[-1](x) 
        return x


## LOW RANK FACTORIZATION VERSION
# class MLP(nn.Module):
#     def __init__(self, args, input_size, hidden_sizes, output_size):
#         super(MLP, self).__init__()
        
#         self.args = args
#         self.kernel_rank = args.aligner_rank

#         layers = []
        
#         if len(hidden_sizes) > 0 :
#             layers.append(nn.Linear(input_size, self.kernel_rank))
#             layers.append(nn.Linear(self.kernel_rank, hidden_sizes[0]))

#             for i in range(1, len(hidden_sizes)):
#                 layers.append(nn.Linear(hidden_sizes[i - 1], self.kernel_rank))
#                 layers.append(nn.Linear(self.kernel_rank, hidden_sizes[i]))

#             layers.append(nn.Linear(hidden_sizes[-1], self.kernel_rank))
#             layers.append(nn.Linear(self.kernel_rank, output_size))

#         else :
#             layers.append(nn.Linear(input_size, self.kernel_rank))
#             layers.append(nn.Linear(self.kernel_rank, output_size))

#         self.layers = nn.ModuleList(layers)
    

#     def forward(self, x):
#         if len(self.layers) == 1 :
#             x = self.layers[0](x)
#             return self.layers[1](x)

#         for i, layer in enumerate(self.layers[:-2]):
#             x = layer(x)
#             if i % 2 == 1 :
#                 x = F.relu(x)  
         
#         x = self.layers[-2](x)  
#         x = self.layers[-1](x) 
#         return x



