import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import time
class SimpleMemory(nn.Module):

    def __init__(self,device, obs_shape,act_shape,dne_shape,batch_size,memory_size, max_sequence_len = 500) -> None:
        
        super(SimpleMemory, self).__init__()

        self.device           = device
        self.memory_size      = torch.Tensor((memory_size,)).to(self.device).to(torch.long)
        self.batch_size       = batch_size
        self.max_sequence_len = max_sequence_len
        self.buffer_size      = max_sequence_len

        self.obs_shape        = obs_shape
        self.act_shape        = act_shape
        self.dne_shape        = dne_shape

        #reserve some space for the memory
        self.obs_mem = torch.zeros( (memory_size,max_sequence_len,obs_shape) ,device= self.device)
        self.act_mem = torch.zeros( (memory_size,max_sequence_len,act_shape) ,device= self.device)
        self.dne_mem = torch.zeros( (memory_size,max_sequence_len,dne_shape) ,device= self.device).to(torch.long)


        #buffers that will be filled at every steps and flushed into memory when long enought or when done
        self.obs_buf = torch.zeros( (self.batch_size,self.buffer_size,obs_shape)  ,device= self.device)
        self.act_buf = torch.zeros( (self.batch_size,self.buffer_size,act_shape)  ,device= self.device)
        self.dne_buf = torch.zeros( (self.batch_size,self.buffer_size,dne_shape)  ,device= self.device).to(torch.long)


        self.buf_indexes      = torch.zeros((self.batch_size),device=self.device).to(torch.long)
        self.mem_index        = torch.Tensor((1,)).to(self.device).to(torch.long)
        self.row_indexes      = torch.arange(0,self.batch_size,device=self.device)

        self.curent_reward    = torch.zeros((1),device=self.device)


    def push(self,O,A,D,seq_len = 30):

        #filling the buffers
        self.obs_buf[self.row_indexes,self.buf_indexes] = O.clone().detach()
        self.act_buf[self.row_indexes,self.buf_indexes] = A.clone().detach()
        self.dne_buf[self.row_indexes,self.buf_indexes] = D.clone().detach()

        #getting the index of the envs buffers that are finished BUT too short to be pushed in memory
        too_short = torch.logical_and(
                                        torch.squeeze(self.dne_buf[self.row_indexes,self.buf_indexes]>0),
                                        self.buf_indexes<seq_len
                                    ).nonzero(as_tuple=True)[0]

        #clearing the buffers of the short ones
        self.obs_buf[too_short]    *= 0
        self.act_buf[too_short]    *= 0
        self.dne_buf[too_short]    *= 0
        #reseting the indexes of the short ones
        self.buf_indexes[too_short] = 0


        #getting the index of the envs buffers that are finished or already too long
        self.buf_indexes+=1
        dones    = torch.logical_or(
                                        torch.squeeze(self.dne_buf[self.row_indexes,self.buf_indexes-1]>0),
                                        self.buf_indexes>=self.buffer_size
                                    ).nonzero(as_tuple=True)[0]
        nb_dones = dones.shape[0]

        if(nb_dones>0):
            
            #getting the rolling memory indexes of where to push the buffers
            mem_indexes = (self.mem_index%self.memory_size) + torch.arange(0,nb_dones,device=self.device).to(torch.long)
            # and mem_index is 9 and nb_dones 5
            # mem_indexes = [9,0,1,2,3,4]
            
            #this is just in case nb_dones > mem_index

            # ex if the curent memory max size is 10
            # and mem_index is 9 and nb_dones 1000
            # mem_indexes = [9,0,1,2,3,4,5,6,7,8,9,10, ... ]
            #                                       OUPS
            test = mem_indexes<self.memory_size
            dones_croped = dones[test]
            mem_indexes  = mem_indexes[test]

            #pushing in the memory
            self.obs_mem[mem_indexes] = self.obs_buf[dones_croped].clone()
            self.act_mem[mem_indexes] = self.act_buf[dones_croped].clone()
            self.dne_mem[mem_indexes] = self.dne_buf[dones_croped].clone()

            self.obs_buf[dones] *= 0
            self.act_buf[dones] *= 0
            self.dne_buf[dones] *= 0

            self.buf_indexes[dones] = 0
            self.mem_index += dones_croped.shape[0]


    def sample(self,batch_size, seq_len = 30):


        # getting the curent index of the memory
        # this is usefull because and the start the memory can be empty so we cant sample from anywhere
        max_index = torch.where(self.mem_index>=self.memory_size,self.mem_index%self.memory_size,self.mem_index)
        if(max_index[0]<=0):
            return None,None,None

        #selected envs to go in the batch
        indexes_batch_env = torch.randint(low =0, high = max_index[0], size = (batch_size,),device=self.device)

        #we add a done in the end of evrery selection just in case the sequence is finished by len and not by done
        select_dne = self.dne_mem[indexes_batch_env]
        dnes_ = torch.ones(select_dne.shape[0],select_dne.shape[1]+1,select_dne.shape[2], device= self.device)
        dnes_[:,0:-1,:] = select_dne

        #we get the index of when done hapened to check where is the last index we can sample from
        dones_indexes    =  torch.argmax((dnes_ > 0).to(torch.long) , dim=1)
        #we remove the ones where we can't sample the good number of steps
        good_len_indexes = ((dones_indexes-seq_len-1)>=0).nonzero(as_tuple=True)[0].to(torch.long)

        indexes_batch_env = indexes_batch_env[good_len_indexes]
        dones_indexes = dones_indexes[good_len_indexes].to(torch.long)

        if(indexes_batch_env.shape[0] == 0):
            return None,None,None
            #TO AVOID THIS WE SHOULD SELECT LENGHT THEN SELECT BATCH
        s = (torch.rand((indexes_batch_env.shape[0],),device=self.device) * (dones_indexes-seq_len)[0]).to(torch.long)
        s = s.unsqueeze(1).repeat(1,seq_len)
        seq_indexes = torch.arange(0,seq_len,device=self.device).unsqueeze(0).repeat(indexes_batch_env.shape[0],1)
        seq_indexes += s
        seq_indexes = seq_indexes.unsqueeze(-1)


        returned_obs = torch.gather(self.obs_mem[indexes_batch_env], 1, seq_indexes.expand(-1, -1, self.obs_mem[indexes_batch_env].shape[-1])).permute((1, 0, 2)).contiguous()
        returned_act = torch.gather(self.act_mem[indexes_batch_env], 1, seq_indexes.expand(-1, -1, self.act_mem[indexes_batch_env].shape[-1])).permute((1, 0, 2)).contiguous()
        returned_dne = torch.gather(self.dne_mem[indexes_batch_env], 1, seq_indexes.expand(-1, -1, self.dne_mem[indexes_batch_env].shape[-1])).permute((1, 0, 2)).contiguous()

        return  (returned_obs,
                 returned_act,
                 returned_dne)

