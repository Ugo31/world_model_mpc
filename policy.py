from torch import nn
import torch
import math
class PI(nn.Module):
    def __init__(self,obs_shape,act_shape,N,K,device):
        
        super(PI, self).__init__()

        self.device = device
        self.lattent_size = 256
        self.K = K
        self.N = N

        print("obs_shape=",obs_shape)
        print("act_shape=",act_shape)

        self.state_embeder  =  nn.Sequential(
                                                    nn.Linear( obs_shape , self.lattent_size),
                                            )
        
        self.act_embeder    =  nn.Sequential(
                                                    nn.Linear( act_shape , self.lattent_size),
                                            )
        
        self.decoder        = nn.Sequential(
                                                    nn.Linear( self.lattent_size , 64),
                                                    nn.GELU(),
                                                    nn.Linear(64, 64),
                                                    nn.GELU(),
                                                    nn.Linear(64, 64),
                                                    nn.GELU(),
                                                    nn.Linear(64, act_shape),
                                                    nn.Tanh()
                                            )

        self.t         = nn.Transformer(d_model=self.lattent_size,nhead = 4,num_encoder_layers=3,num_decoder_layers=3,dim_feedforward = 256,dropout=0)
        self.pos_enc   = PositionalEncoding(d_model=self.lattent_size,max_len = 40,dropout=0)




    def forward(self,O):


        decoder_input = torch.ones(size=(self.K, O.shape[1],O.shape[2]), device=self.device) 
        decoder_input = decoder_input*O[-1]

        x_encoded,y_encoded = self.embed_(O,decoder_input)
        
        encoder_output    = self.t.encoder(x_encoded)
        decoder_output = self.t.decoder(y_encoded, encoder_output)
        
        y = self.decoder(decoder_output)


        #print(y.shape)

        return y
    





    def embed_(self,O,Y = None):
        if(O is not None):
        
            ZO = self.state_embeder(O)
            x_encoded = self.pos_enc(ZO)
        else:
            x_encoded = None

        if(Y is not None):
            ZY = self.state_embeder(Y)
            y_encoded = self.pos_enc(ZY)
        else:
            y_encoded = None
        return x_encoded,y_encoded



    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size,device=self.device) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x#self.dropout(x)
