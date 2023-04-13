# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 00:00:53 2023

@author: hrish
"""




import numpy as np
import random
import copy
from matplotlib import pyplot as plt

sim_time = 2000000
# sim_time = 100
c_size = 50                 #in meter
# w_speed = 20000/60          #in m/s
w_speed = 2000/60          #in m/s
Ts = 0.01                   #in sec
# P_c = 0.73
P_c = 0.20

P_charge = 0.9              #Charging Probbaility when energy is harvested
B = 150                      #Battery Capacity (in Pkts tx capability) For reception: Tx/1000


mu_c = c_size/w_speed

mu_s = c_size*(1-P_c)/(w_speed*P_c)


R_1_2 = (Ts/mu_s)*np.exp(Ts/mu_s)
R_2_1 = (Ts/mu_c)*np.exp(Ts/mu_c)

R_1_1 = 1-R_1_2
R_2_2 = 1-R_2_1


cur_st = 0
b_status = [0]

harv_enr = []







num_flows=1
out_4_rl=1

queue_max = 1000

queue_length = [0]
pkt_miss = [0]
in_pkt = [0]
out_pkt = [0]
eff_in_pkt = [0]

queue_loss = [0]
pkt_miss_energy = [0]
pkt_miss_intent = [0]

pkt_entry = [0]

true_on = [0]


def moving_average(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n









def network(af, af_t, epochid, q_f, est_lam, pkts, plcy):
    
    
    
    global P_c, P_charge, B, R_1_2, R_2_1,R_1_1, R_2_2,cur_st,b_status,harv_enr, queue_length, pkt_miss, in_pkt, true_on, pkt_entry
    
    global time
    
    p_l = 0
    p_m = 0
    p_n = 0
    p_o = 0
    
    
    
    p_i_ef = 0
    
    on = 0
    
    q_loss = 0
    
    tx_p = np.zeros((num_flows,1))
    on_p = np.zeros((num_flows,1))
    # prev_qf = np.zeros((num_flows,1))
    pkt_in = np.zeros((num_flows,1))
    vf_cnt = np.zeros((num_flows,1))
    
    
    
    prev_qf = copy.deepcopy(q_f)

    
    # q_f1 = q_f[0]
    # q_f2 = q_f[1]
    
    # prev_qf1 = copy.deepcopy(q_f1)
    # prev_qf2 = copy.deepcopy(q_f2)
    
    # pkt_in1 = 0
    # pkt_in2 = 0
    
    
    for iter in range(num_flows):
        tx_p[iter][0]=af_t[iter]/20
        on_p[iter][0]=af[iter]/20
    
    
    
    act = [[] for _ in range(num_flows)]
    act_t = [[] for _ in range(num_flows)]
    p4j = [[] for _ in range(num_flows)]
    lam_hat = [[] for _ in range(num_flows)]
    diff = [[] for _ in range(num_flows)]
    l_state = [[] for _ in range(num_flows)]
    rew = [[] for _ in range(num_flows)]
    
    
    
    
    # frame_cnt = 0
    
    # p4j1 = []
    # p4j2 = []
    
    
    # vf_cnt1 = 0    
    # vf_cnt2 = 0
 
    if plcy == 'S':
 
        for i in range(out_4_rl):            
            
            
            if cur_st == 0:
                if np.random.rand()<=R_2_1:
                    nxt_st = 1
                else:
                    nxt_st = 0
                harv_enr.append(0)
                bat_level = b_status[-1]
            else:
                if np.random.rand()<=R_1_2:
                    nxt_st = 0
                else:
                    nxt_st = 1
                harv_enr.append(1)
                
                if np.random.rand()<=P_charge and b_status[-1]<B:
                    bat_level = b_status[-1]+1
                else:
                    bat_level = b_status[-1]
                    
                
                
            cur_st = copy.deepcopy(nxt_st)
            
            
            flow_active_list = []
            
            tmp_ind = i
            
            for iter in range(num_flows):
                if np.random.rand()<=tx_p[iter][0]:
                    act[iter] = 1
                    flow_active_list.append(iter)
                else:
                    act[iter] = 0 
                
                
                # if b_status[-1]>0:
                #     on+=1
                
                if queue_length[-1]>=queue_max:
                    q_loss+=int(pkts[iter][0][tmp_ind])
                    p_n+=int(pkts[iter][0][tmp_ind])
                else:
                    p_l+=int(pkts[iter][0][tmp_ind])
                    if b_status[-1]>=0.04 and act[iter] == 1:
                        on+=1
                        if (q_f[iter]+int(pkts[iter][0][tmp_ind])>queue_max):
                            q_f[iter] = queue_max
                            q_loss+=(q_f[iter]+int(pkts[iter][0][tmp_ind])-queue_max)
                        else:
                            q_f[iter] += int(pkts[iter][0][tmp_ind])
                        pkt_in[iter]+=int(pkts[iter][0][tmp_ind])
                        # bat_level=bat_level-0.047
                        bat_level=bat_level-0.04 ## 20m communication range
                        p_n+=int(pkts[iter][0][tmp_ind])
                        p_i_ef+=int(pkts[iter][0][tmp_ind])
                    else:
                        p_m+=int(pkts[iter][0][tmp_ind])
                        p_n+=int(pkts[iter][0][tmp_ind])
                
            if sum(act)>1:
                ac_flow_in = random.randint(0, len(flow_active_list)-1)
                ac_flow = flow_active_list[ac_flow_in]
                for c in range(num_flows):
                    if c!= ac_flow:
                        act[c]=0
            
    
            
            for iter in range(num_flows):
                if act[iter]==1 and q_f[iter]>0 and b_status[-1]>=1:
                    q_f[iter]-=1
                    vf_cnt[iter]+=1
                    p4j[iter].append(1)
                    bat_level=bat_level-1
                    p_o+=1
                else:
                    p4j[iter].append(0)
    

 

    elif plcy == 'T':
 
        for i in range(out_4_rl):            
            
            
            if cur_st == 0:
                if np.random.rand()<=R_2_1:
                    nxt_st = 1
                else:
                    nxt_st = 0
                harv_enr.append(0)
                bat_level = b_status[-1]
            else:
                if np.random.rand()<=R_1_2:
                    nxt_st = 0
                else:
                    nxt_st = 1
                harv_enr.append(1)
                
                if np.random.rand()<=P_charge and b_status[-1]<B:
                    bat_level = b_status[-1]+1
                else:
                    bat_level = b_status[-1]
                    
                
                
            cur_st = copy.deepcopy(nxt_st)
            
            
            flow_active_list = []
            
            tmp_ind = copy.deepcopy(time)
            
            for iter in range(num_flows):
                if np.random.rand()<=tx_p[iter][0]:
                    act[iter] = 1
                    flow_active_list.append(iter)
                else:
                    act[iter] = 0 
                
                
                # if b_status[-1]>0:
                #     on+=1
                
                if queue_length[-1]>=queue_max:
                    q_loss+=int(pkts[iter][0][tmp_ind])
                    p_n+=int(pkts[iter][0][tmp_ind])
                else:
                    p_l+=int(pkts[iter][0][tmp_ind])
                    if b_status[-1]>=0.04:
                        on+=1
                        if (q_f[iter]+int(pkts[iter][0][tmp_ind])>queue_max):
                            q_f[iter] = queue_max
                            q_loss+=(q_f[iter]+int(pkts[iter][0][tmp_ind])-queue_max)
                        else:
                            q_f[iter] += int(pkts[iter][0][tmp_ind])
                        pkt_in[iter]+=int(pkts[iter][0][tmp_ind])
                        # bat_level=bat_level-0.047
                        bat_level=bat_level-0.04 ## 20m communication range
                        p_n+=int(pkts[iter][0][tmp_ind])
                        p_i_ef+=int(pkts[iter][0][tmp_ind])
                    else:
                        p_m+=int(pkts[iter][0][tmp_ind])
                        p_n+=int(pkts[iter][0][tmp_ind])
                
            if sum(act)>1:
                ac_flow_in = random.randint(0, len(flow_active_list)-1)
                ac_flow = flow_active_list[ac_flow_in]
                for c in range(num_flows):
                    if c!= ac_flow:
                        act[c]=0
            
    
            
            for iter in range(num_flows):
                if act[iter]==1 and q_f[iter]>0 and b_status[-1]>=1:
                    q_f[iter]-=1
                    vf_cnt[iter]+=1
                    p4j[iter].append(1)
                    bat_level=bat_level-1
                    p_o+=1
                else:
                    p4j[iter].append(0)
                    
                    
                    
                    
                    
                    
                    
                    
    elif plcy == 'ST':
 
        for i in range(out_4_rl):            
            
            
            if cur_st == 0:
                if np.random.rand()<=R_2_1:
                    nxt_st = 1
                else:
                    nxt_st = 0
                harv_enr.append(0)
                bat_level = b_status[-1]
            else:
                if np.random.rand()<=R_1_2:
                    nxt_st = 0
                else:
                    nxt_st = 1
                harv_enr.append(1)
                
                if np.random.rand()<=P_charge and b_status[-1]<B:
                    bat_level = b_status[-1]+1
                else:
                    bat_level = b_status[-1]
                    
                
                
            cur_st = copy.deepcopy(nxt_st)
            
            
            flow_active_list = []
            
            tmp_ind = copy.deepcopy(time)
            
            for iter in range(num_flows):
                if np.random.rand()<=on_p[iter][0]:
                    act[iter] = 1
                else:
                    act[iter] = 0
                    
                if np.random.rand()<=tx_p[iter][0]:
                    act_t[iter] = 1
                    flow_active_list.append(iter)
                else:
                    act_t[iter] = 0
                
                
                # if b_status[-1]>0:
                #     on+=1
                
                
                if b_status[-1]>=0.04:
                    if q_f[iter]>=queue_max:
                        # print('queue_max:',queue_max)
                        # print(q_f[iter],int(pkts[iter][0][tmp_ind]))
                        q_loss+=int(pkts[iter][0][tmp_ind])
                        p_n+=int(pkts[iter][0][tmp_ind])
                    else:
                        p_l+=int(pkts[iter][0][tmp_ind])
                        if act[iter] == 1:
                            on+=1
                            if (q_f[iter]+int(pkts[iter][0][tmp_ind])>queue_max):
                                q_f[iter] = queue_max
                                q_loss+=(q_f[iter]+int(pkts[iter][0][tmp_ind])-queue_max)
                            else:
                                q_f[iter] += int(pkts[iter][0][tmp_ind])
                            
                            pkt_in[iter]+=int(pkts[iter][0][tmp_ind])
                            # bat_level=bat_level-0.047
                            bat_level=bat_level-0.04 ## 20m communication range
                            p_n+=int(pkts[iter][0][tmp_ind])
                            p_i_ef+=int(pkts[iter][0][tmp_ind])
                        else:
                            p_m+=int(pkts[iter][0][tmp_ind])
                            p_n+=int(pkts[iter][0][tmp_ind])
                else:
                    p_m+=int(pkts[iter][0][tmp_ind])
                    p_n+=int(pkts[iter][0][tmp_ind])
                
                
                
                
                # if q_f[iter]>=queue_max:
                #     q_loss+=int(pkts[iter][0][tmp_ind])
                #     p_n+=int(pkts[iter][0][tmp_ind])
                # else:
                #     p_l+=int(pkts[iter][0][tmp_ind])
                #     if b_status[-1]>0 and act[iter] == 1:
                #         on+=1
                #         q_f[iter] += int(pkts[iter][0][tmp_ind])
                #         pkt_in[iter]+=int(pkts[iter][0][tmp_ind])
                #         # bat_level=bat_level-0.047
                #         bat_level=bat_level-0.04 ## 20m communication range
                #         p_n+=int(pkts[iter][0][tmp_ind])
                #         p_i_ef+=int(pkts[iter][0][tmp_ind])
                #     else:
                #         p_m+=int(pkts[iter][0][tmp_ind])
                #         p_n+=int(pkts[iter][0][tmp_ind])
                
            if sum(act_t)>1:
                ac_flow_in = random.randint(0, len(flow_active_list)-1)
                ac_flow = flow_active_list[ac_flow_in]
                for c in range(num_flows):
                    if c!= ac_flow:
                        act_t[c]=0
            
    
            
            for iter in range(num_flows):
                if act[iter]==1 and act_t[iter]==1 and q_f[iter]>0 and b_status[-1]>=1:
                    q_f[iter]-=1
                    vf_cnt[iter]+=1
                    p4j[iter].append(1)
                    bat_level=bat_level-1
                    p_o+=1
                else:
                    p4j[iter].append(0)
                    
                    
                    
                    
                    
    

    


    for iter in range(num_flows):
        lam_hat[iter] = (((q_f[iter]-prev_qf[iter])+vf_cnt[iter])/100)[0]
        diff[iter] = int (q_f[iter]-prev_qf[iter])
        
        if diff[iter]>0:
            l_state[iter] = 0
        if diff[iter]==0:
            l_state[iter] = 1
        if diff[iter]<0:
            l_state[iter] = 2
            
        if af[iter]/10>est_lam[iter] and (af[iter]/10)-est_lam[iter]<0.15:
            rew[iter] = 1.0
        else:
            rew[iter] = -1.0
    
    b_status.append(bat_level)
    pkt_miss.append(p_m)
    in_pkt.append(p_n)
    out_pkt.append(p_o)
    eff_in_pkt.append(p_i_ef)
    queue_length.append(q_f[0])
    true_on.append(on)
    queue_loss.append(q_loss)
    pkt_entry.append(p_l)
    
    return l_state, rew, q_f, p4j, pkt_in, lam_hat




















    
    
p_rate = [2,4,6,8,10,12,14,16,18,20]

tx_rate = [2,4,6,8,10,12,14,16,18,20]  



# p_rate = [10]

# tx_rate = [8,10,12,14]

# p_rate = [2,10,20]    
    
metric = []    
metric_array = np.empty((len(p_rate),len(tx_rate),6))
 

pkt_rate = 0.75
policy = 'ST'
    
 
for in_p,p in enumerate(p_rate):
    
    for in_tx_p, tx_p in enumerate(tx_rate):
    
        queue_length = [0]
        pkt_miss = [0]
        in_pkt = [0]
        out_pkt = [0]
        eff_in_pkt = [0]
    
        queue_loss = [0]
        pkt_miss_energy = [0]
        pkt_miss_intent = [0]
    
        pkt_entry = [0]
    
        true_on = [0]
        
        rate = [0.20 for _ in range(num_flows)]
        ns = 0.0
        q = [0]
        
        q_src = 0
        
        
        
        
        
        p2t = []
        p2t_src = []
        
        for c in range(num_flows):
            p2t.append(np.zeros((1,sim_time)))
        
        for c in range(num_flows):
            p2t[c] = np.array([np.random.poisson(lam=pkt_rate, size=sim_time).tolist()])
            
            for x_it in p2t[c][0]:
                if x_it == 0:
                    p2t_src.append(min(q_src,1))
                    q_src = max(q_src-1,0)
                else:
                    p2t_src.append(min(x_it,1))
                if x_it>1:
                    q_src += x_it-1
            for c in range(num_flows):
                p2t[c] = copy.deepcopy(np.reshape(np.array(p2t_src),(1,sim_time)))
        
        
            
     
        for time in range(sim_time):
            
            
            
            action = np.array([p])
            

                
                
                
            
            ns, reward, q, pd, p_in, est_rate = network(action, np.array([tx_p]), time,q, rate, p2t, policy)
            
            for c in range(num_flows):
                rate[c] = 0.5*(rate[c]+est_rate[c])
                
                
            if time%1000000==0:
                print(in_p, in_tx_p)
                    
            
        
        miss_rate = [y*100/in_pkt[x] for x, y in enumerate(pkt_miss) if in_pkt[x]!=0]
        
        queue_loss_rate = [y*100/in_pkt[x] for x, y in enumerate(queue_loss) if in_pkt[x]!=0]
        
        miss_rate_2 = [y*100/pkt_entry[x] for x, y in enumerate(pkt_miss) if pkt_entry[x]!=0]
    
    
        # print('Average Packet Miss rate', np.mean(miss_rate),'%')
        # print('Average queue length', np.mean(queue_length))
        # print('Average Effective lambda', np.mean(eff_in_pkt))
        # print('Average Effective mu', np.mean(out_pkt))
        # print('True on:', np.mean(true_on)*100,'%')
        # print('Queue overflow:', np.mean(queue_loss_rate)*100,'%')
        
        metric_array[in_p][in_tx_p][0]=np.mean(miss_rate)
        metric_array[in_p][in_tx_p][1]=np.mean(true_on)*100
        metric_array[in_p][in_tx_p][2]=np.mean(queue_length)
        metric_array[in_p][in_tx_p][3]=np.mean(queue_loss_rate)
        metric_array[in_p][in_tx_p][4]=np.mean(queue_loss)
        metric_array[in_p][in_tx_p][5]=np.mean(miss_rate_2)
        
        metric_list = [np.mean(miss_rate),np.mean(true_on)*100,np.mean(queue_length),np.mean(queue_loss), np.mean(eff_in_pkt),np.mean(miss_rate_2)]
        
        metric.append(copy.deepcopy(metric_list))





    