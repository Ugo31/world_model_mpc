import isaacgym
import isaacgymenvs
import torch
assert torch.cuda.is_available()

import torch.multiprocessing as multy_process
torch.multiprocessing.set_start_method('spawn', force=True)

import traceback
import logging
from memory import SimpleMemory
from matplotlib import pyplot as plt
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
from isaacgymenvs.tasks.cartpole import compute_reward

from world_model import WM
from policy      import PI

np.set_printoptions(linewidth = 150)


N                 = 10
K                 = 20

MEMORY_LEN        = 20000
OPT_LIMIT         = 0.0
TRAIN_WM          = True
TRAIN_OPT         = False

n_envs            = 512



def use_grad(model,bool_):
    for param in model.parameters():
        param.requires_grad = bool_




def update_WM(mem,model,opt,batch_size,device,path):
    writer = SummaryWriter(path)

    use_grad(model,True)
    model.train()
    worldmodel_epoch = 0
    while True:
        verbose = worldmodel_epoch%100 == 0
        timer_ = time.time()
        L_Ot,L_At,L_Dt = mem.sample(batch_size,seq_len=K+N)
        if(L_Ot is None):
            pass
        else:
            
            NOt   = L_Ot[:N]
            NKAt  = L_At[N:N+K]
            NKOt  = L_Ot[N:N+K]

            pred_F2 = model.forward(NOt,NKAt)
            d0_loss  = torch.mean((pred_F2-NKOt)**2) 

            loss_wmd = d0_loss
            opt.zero_grad()
            loss_wmd.backward()
            opt.step()

            if(verbose == 1):

                plt.figure(1)
                plt.subplot(151)
                plt.cla()
                o_index = 0
                plt.figure(1)
                plt.plot(np.arange(0,N),NOt[:,0,o_index].cpu().detach().numpy(), color = "blue")
                plt.plot(np.arange(N,K+N),NKOt[:,0,o_index].cpu().detach().numpy(), color = "green")
                plt.plot(np.arange(N,K+N),pred_F2[:,0,o_index].cpu().detach().numpy(), color = "red")

                plt.subplot(152)
                o_index = 1
                plt.cla()
                plt.plot(np.arange(0,N),NOt[:,0,o_index].cpu().detach().numpy(), color = "blue")
                plt.plot(np.arange(N,K+N),NKOt[:,0,o_index].cpu().detach().numpy(), color = "green")
                plt.plot(np.arange(N,K+N),pred_F2[:,0,o_index].cpu().detach().numpy(), color = "red")
                plt.subplot(153)
                o_index = 2
                plt.cla()
                plt.plot(np.arange(0,N),NOt[:,0,o_index].cpu().detach().numpy(), color = "blue")
                plt.plot(np.arange(N,K+N),NKOt[:,0,o_index].cpu().detach().numpy(), color = "green")
                plt.plot(np.arange(N,K+N),pred_F2[:,0,o_index].cpu().detach().numpy(), color = "red")
                plt.subplot(154)
                o_index = 3
                plt.cla()
                plt.plot(np.arange(0,N),NOt[:,0,o_index].cpu().detach().numpy(), color = "blue")
                plt.plot(np.arange(N,K+N),NKOt[:,0,o_index].cpu().detach().numpy(), color = "green")
                plt.plot(np.arange(N,K+N),pred_F2[:,0,o_index].cpu().detach().numpy(), color = "red")
                plt.subplot(155)
                plt.cla()
                plt.plot(NKAt[:,0,0].cpu().detach().numpy(), color = "blue")
                plt.draw()
                plt.pause(0.1)

            worldmodel_epoch+=1
            dt = (time.time()-timer_)
            print("Model updates:\t",worldmodel_epoch,"\t| dt :\t",dt)
            writer.add_scalar("MODEL/loss",   loss_wmd.item(), worldmodel_epoch)
            if(worldmodel_epoch%10000 ==0):
                torch.save(model.state_dict(),"./models/transformer_step_"+str(worldmodel_epoch)+"_pytroch")

def update_PI(mem,model,wm,opt,batch_size,reset_dist,device,path):
        writer = SummaryWriter(path)
        policy_epoch = 0

        use_grad(wm,False)
        wm.eval()

        while True:
            timer_ = time.time()
            torch.cuda.synchronize(device=device)

            verbose = policy_epoch%100 == 0
            L_Ot,_,_ = mem.sample(batch_size,seq_len=K+N)

            if(L_Ot is not None):

                NOt   = L_Ot[:N].detach()
                pred_seq  = model(NOt)

                opti_seq  = pred_seq.clone().detach()
                opti_seq.requires_grad=True
                optimizer   = torch.optim.SGD([opti_seq], lr=1)

                rwd_opt     = []
                futures_opt = []
                act_opt     = []
                for _ in range(1):

                    pred_obs    = wm.forward(NOt,opti_seq)
                    pole_angle  = pred_obs[:,:, 2]
                    pole_vel    = pred_obs[:,:, 3]
                    cart_vel    = pred_obs[:,:, 1]
                    cart_pos    = pred_obs[:,:, 0]
                    rwd    = compute_reward(pole_angle, pole_vel, cart_vel, cart_pos, reset_dist)

                    noise = torch.sum((opti_seq[1:]-opti_seq[:-1])**2)
                    limits_loss = torch.exp(torch.pow(opti_seq,30))-1
                    mean_rwd = torch.mean(rwd)
                    a_loss = -mean_rwd + torch.sum(limits_loss) #+ noise

                    optimizer.zero_grad()
                    a_loss.backward()
                    optimizer.step()
                    
                    if(verbose):
                        rwd_opt.append(mean_rwd.item())
                        futures_opt.append(pred_obs[:,0,2])
                        act_opt.append(opti_seq[:,0,0].clone())


                opti_seq = opti_seq.detach().clone()
                importance_factor = torch.arange(K,0,-1,device=device)/K
                nn_loss   = torch.mean(((opti_seq-pred_seq)*importance_factor)**2)
                opt.zero_grad()
                nn_loss.backward()
                opt.step()
                

                if(verbose):
                    plt.figure(2)
                    plt.subplot(141)
                    plt.cla()
                    plt.gca().set_title('Rewards')
                    plt.plot(rwd_opt)

                    plt.subplot(142)
                    plt.cla()
                    plt.gca().set_title('Observations')
                    plt.gca().set_ylim([-2, 2])
                    plt.plot(NOt[:,0,2].detach().cpu().numpy())
                    index = 0
                    for i in futures_opt:
                        b = (index+1)/len(futures_opt)
                        plt.plot(np.arange(N,N+K),i.detach().cpu().numpy(),color = (0.2,b,b))
                        index+=1
                    plt.draw()

                    plt.subplot(143)
                    plt.cla()
                    plt.gca().set_title('Actions')
                    index = 0
                    for i in act_opt:
                        b = (index+1)/len(act_opt)
                        plt.plot(i.detach().cpu().numpy(),color = (b,0.2,b))
                        index+=1
                    plt.plot(pred_seq[:,0,0].detach().cpu().numpy(), color = "green")
                    plt.draw()
                    plt.pause(0.1)
                

                if(policy_epoch%1000 ==0):
                    torch.save(model.state_dict(),"./models/policy_step_"+str(policy_epoch)+"_pytroch")


                policy_epoch+=1
                dt = (time.time()-timer_)
                print("Policy Train:\t",policy_epoch,"\t| dt :\t",dt)
                writer.add_scalar("POLICY/loss",          nn_loss.item(),    policy_epoch*batch_size)
                writer.add_scalar("POLICY/reward_update", mem.curent_reward.item(), policy_epoch*batch_size)

def run_policy(mp,mem,envs,device, path):

    writer = SummaryWriter(path)

    roling_Ot       = torch.zeros(N,n_envs,envs.observation_space.shape[0], device=device)
    At              = torch.randn((n_envs,envs.action_space.shape[0]), device=device)
    epsilon = 1.0
    decay   = 0.99
    step_number = 0

    total_time  = 0

    while True:

        timer_ = time.time()
        use_grad(mp,False)
        mp.eval()
        obs, rwd , done, _  = envs.step(At)
        Ot                  = obs['obs'].clone()
        Dt                  = done.view(-1,1).clone()
        mem.push(Ot,At,Dt,seq_len=K+N)

        roling_Ot[:-1]      = roling_Ot[1:].clone()
        roling_Ot[-1]       = Ot

        r                   = torch.rand((n_envs,envs.action_space.shape[0]), device=device)
        rAt                 = At + (torch.randn(At.shape, device=device)/10)
        nAt                 = mp(roling_Ot)[0]
        At                  = torch.where(r>epsilon,nAt,rAt) 

        epsilon            *= decay
        if(epsilon<0.05):
            epsilon = 0.00

        step_number+=1
        dt = time.time()-timer_
        total_time+=dt

        if(step_number>10):
            mean_reward = torch.sum(rwd)
            mem.curent_reward[0] = mean_reward.item()
            print("Policy Step:\t",step_number,"\t| dt :\t",dt)
            writer.add_scalar("POLICY/reward_step", mean_reward, step_number*n_envs)
            writer.add_scalar("POLICY/reward_time", mean_reward, total_time)
            writer.add_scalar("POLICY/epsilon", epsilon, step_number*n_envs)

        if(step_number*512 > 2e6):
            break

def main():

    path = "./runs/"+str(time.time())

    #######################  ENV INIT  #######################
    device = "cuda:0"
    envs = isaacgymenvs.make(
                                seed=0, 
                                task="Cartpole", 
                                num_envs=n_envs, 
                                sim_device=device,
                                rl_device=device,
                                graphics_device_id=0,
                                headless=False
                            )

    #######################  MODELS INIT  #######################
    wm = WM(

                lattent_size = 64 ,
                obs_shape = envs.observation_space.shape[0],
                act_shape = envs.action_space.shape[0],
                K=K,
                N=N,
                device=device,

            ).to(device)
    #can be even faster if you uncoment this
    #wm.load_state_dict(torch.load("./models/model"))
    mp = PI(
                envs.observation_space.shape[0],
                envs.action_space.shape[0],
                K=K,
                N=N,
                device=device,
            ).to(device)


    mem = SimpleMemory(
                            device = device,
                            obs_shape = envs.observation_space.shape[0],
                            act_shape=envs.action_space.shape[0],
                            dne_shape = 1,
                            batch_size=n_envs,
                            memory_size=20000,
                            max_sequence_len = 200
                        ).to(device)

    wm.share_memory()
    mp.share_memory()
    mem.share_memory()

    try:

        opt_mp   = torch.optim.Adam(mp.parameters(), lr=1e-4,amsgrad=True)
        opt_wm   = torch.optim.Adam(wm.parameters(), lr=1e-4,amsgrad=True)


        p_wm = multy_process.Process(target=update_WM, args=(mem,wm,opt_wm,512,device,path))
        p_wm.start()
        p_mp = multy_process.Process(target=update_PI, args=(mem,mp,wm,opt_mp,32,envs.reset_dist,device,path))
        p_mp.start()
        run_policy(mp,mem,envs,device,path)

    except Exception as e:
        print("==KILLING PROCESSES==")
        p_wm.kill()
        p_mp.kill()
        p_mp.join()
        p_mp.join()
        logging.error(traceback.format_exc())






if __name__ == '__main__':
    main()
