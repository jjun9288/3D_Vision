import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Positional encoding
def positional_encoding(position, n_freq, include_input=False) :

    freq_bands = 2.**(torch.linspace(0, n_freq-1, n_freq))
    
    pe_cos = torch.cat([torch.cos(freq_band * torch.pi * position) for freq_band in freq_bands], dim=-1)
    pe_sin = torch.cat([torch.sin(freq_band * torch.pi * position) for freq_band in freq_bands], dim=-1)

    if include_input : 
        return(torch.cat((position, pe_cos, pe_sin), dim=-1))
    
    return(torch.cat((pe_cos, pe_sin), dim=-1))



# Model implementation
class NeRF(nn.Module) : 
    
    def __init__(self, num_pos, num_view) : 
        super().__init__()
        #self.num_pos = num_pos
        #self.num_view = num_view

        self.FC1 = nn.Linear(num_pos, 256)
        self.FC2 = nn.Linear(256, 256)
        self.FC3 = nn.Linear(256, 256)
        self.FC4 = nn.Linear(256, 256)
        self.FC5 = nn.Linear(256 + num_pos, 256)
        self.FC6 = nn.Linear(256, 256)
        self.FC7 = nn.Linear(256, 256)
        self.FC8 = nn.Linear(256, 256)
        self.sigma_layer = nn.Linear(256, 1)
        self.feature_layer = nn.Linear(256, 256)
        self.FC9 = nn.Linear(256 + num_view, 256)
        self.FC10 = nn.Linear(256, 128)
        self.RGB_layer = nn.Linear(128, 3)

    def forward(self, pos, view_dir) :
        '''
        - inputs
        x : Batches of embedded 3D positions  
        d : Batches of embedded viewing directions

        - outputs
        sigma : density
        RGB : pixel color
        '''

        x = self.FC1(pos)
        x = F.relu(x)
        x = self.FC2(x)
        x = F.relu(x)
        x = self.FC3(x)
        x = F.relu(x)
        x = self.FC4(x)
        x = F.relu(x)
        x = torch.cat((x, pos), dim=-1)
        x = self.FC5(x)
        x = F.relu(x)
        x = self.FC6(x)
        x = F.relu(x)
        x = self.FC7(x)
        x = F.relu(x)
        x = self.FC8(x)
        x = F.relu(x)
        sigma = self.sigma_layer(x)

        #d = self.feature_layer(x)
        x = torch.cat([x, view_dir], dim=-1)
        x = self.FC9(x)
        x = F.relu(x)
        x = self.FC10(x)
        x = F.relu(x)
        rgb = self.RGB_layer(x)

        return sigma, rgb



# Model test
x = torch.rand(3)
d = torch.rand(3)

x = positional_encoding(x, 10, include_input=True)
d = positional_encoding(d, 4, include_input=True)

model = NeRF(len(x), len(d))
out = model(x, d)
print(out)