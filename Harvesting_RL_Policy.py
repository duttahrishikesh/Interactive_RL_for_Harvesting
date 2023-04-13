# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 19:10:44 2023

@author: hrish
"""


import numpy as np
import copy
# from matplotlib import pyplot as plt
import math
from random import randint
from random import random





# sim_time = 500000
sim_time = 200000
c_size = 50                 #in meter
# w_speed = 20000/60          #in m/s
w_speed = 2000/60          #in m/s
Ts = 0.01                   #in sec
P_c = 0.20

P_charge = 0.9              #Charging Probability when energy is harvested
B = 150                      #Battery Capacity (in Pkts tx capability) For reception: Tx/1000


mu_c = c_size/w_speed

mu_s = c_size*(1-P_c)/(w_speed*P_c)


R_1_2 = (Ts/mu_s)*np.exp(Ts/mu_s)
R_2_1 = (Ts/mu_c)*np.exp(Ts/mu_c)

R_1_1 = 1-R_1_2
R_2_2 = 1-R_2_1









# num_flows=1
# out_4_rl=1

queue_max = 1000




rew_norm_fact = 100/queue_max






history = 200                #in number of frames
num_states = 10
num_action_s = 10
num_action_t = 10




eps_max = 1.0
exp_decay = 3000.0
gamma = 0.99
alpha = 0.1




avg_strt = 150000




def moving_average(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n









def network(af, af_t, epochid, q_f, prev_miss, pkts, plcy,n_flow, act_fl, is_src):
    
    
    
    global P_c, P_charge, B, R_1_2, R_2_1,R_1_1, R_2_2,cur_st,b_status,harv_enr, queue_length, pkt_miss, in_pkt, true_on, pkt_entry, queue_max
    global num_action_s, num_action_t, history    
    global priority, rew_norm_fact
    global p4j
    
    p_l = 0
    p_m = 0
    p_n = 0
    p_o = 0
    
    
    
    p_i_ef = 0
    
    on = 0
    
    q_loss = 0
    
    tx_p = np.zeros((n_flow,1))
    on_p = np.zeros((n_flow,1))
    # prev_qf = np.zeros((num_flows,1))
    pkt_in = np.zeros((n_flow,1))
    
    
    

    
    
    for iter in range(n_flow):
        tx_p[iter][0]=af_t[0]/num_action_t
        on_p[iter][0]=af[0]/num_action_s
    
    
    
    act = [[] for _ in range(n_flow)]
    act_t = [[] for _ in range(n_flow)]
    
    l_state = [[] for _ in range(n_flow)]
    l_state_sleep = [[] for _ in range(n_flow)]
    rew1 = [[] for _ in range(n_flow)]
    rew2 = [[] for _ in range(n_flow)]
    
    
    

 
    if plcy == 'RL':
 
        for i in range(history):            
            
            
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
            
            for iter in range(n_flow):
                if np.random.rand()<=on_p[iter][0]:
                    act[iter] = 1
                else:
                    act[iter] = 0
                    
                if np.random.rand()<=tx_p[iter][0]:
                    act_t[iter] = 1
                    flow_active_list.append(iter)
                else:
                    act_t[iter] = 0
                
                
                if is_src:
                    if q_f[iter][0]>=queue_max:
                        # print('queue_max:',queue_max)
                        # print(q_f[iter],int(pkts[iter][0][tmp_ind]))
                        q_loss+=int(pkts[iter][0][tmp_ind])
                        p_n+=int(pkts[iter][0][tmp_ind])
                    else:
                        p_l+=int(pkts[iter][0][tmp_ind])
                        on+=1
                        if (q_f[iter][0]+int(pkts[iter][0][tmp_ind])>queue_max):
                            q_f[iter][0] = queue_max
                            q_loss+=(q_f[iter][0]+int(pkts[iter][0][tmp_ind])-queue_max)
                        else:
                            q_f[iter][0] += int(pkts[iter][0][tmp_ind])
                        pkt_in[iter]+=int(pkts[iter][0][tmp_ind])
                        # bat_level=bat_level-0.047
                        bat_level=bat_level-0.04 ## 20m communication range
                        p_n+=int(pkts[iter][0][tmp_ind])
                        p_i_ef+=int(pkts[iter][0][tmp_ind])
                else:
                    if b_status[-1]>=0.04:
                        if q_f[iter][0]>=queue_max:
                            # print('queue_max:',queue_max)
                            # print(q_f[iter],int(pkts[iter][0][tmp_ind]))
                            q_loss+=int(pkts[iter][0][tmp_ind])
                            p_n+=int(pkts[iter][0][tmp_ind])
                        else:
                            p_l+=int(pkts[iter][0][tmp_ind])
                            if act[iter] == 1:
                                on+=1
                                if (q_f[iter][0]+int(pkts[iter][0][tmp_ind])>queue_max):
                                    q_f[iter][0] = queue_max
                                    q_loss+=(q_f[iter][0]+int(pkts[iter][0][tmp_ind])-queue_max)
                                else:
                                    q_f[iter][0] += int(pkts[iter][0][tmp_ind])
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
                
            if sum(act_t)>1:
                ac_flow_in = randint(0, len(flow_active_list)-1)
                ac_flow = flow_active_list[ac_flow_in]
                for c in range(n_flow):
                    if c!= ac_flow:
                        act_t[c]=0
            
    
            
            for iter in range(n_flow):
                if act[iter]==1 and act_t[iter]==1 and q_f[iter][0]>0 and b_status[-1]>=1:
                    q_f[iter][0]-=1
                    p4j[iter].append(1)
                    bat_level=bat_level-1
                    p_o+=1
                else:
                    p4j[iter].append(0)
    

 




                    
    

    
    
    
    

    
        
    l_state= np.min([round(np.mean(harv_enr[-history:])*10),9])
    
    l_state_sleep = af_t[0]
    
        
    # if p_m<=prev_miss and q_f[0]<=prev_qf[0]:
    #     rew[iter] = 1.0
    # else:
    #     rew[iter] = -1.0
    
    q_s = 0
    for q_it in range(n_flow):
        q_s = q_s+q_f[q_it][0]
        queue_length[q_it].append(q_f[q_it][0])
    
    
    if q_s<0.9*queue_max*n_flow:
        # rew1[iter]= -200*(p_m)
        # rew2[iter]= -200*(p_m)
        rew1= 100/(p_m+0.001)
        rew2= 100/(p_m+0.001)
        # rew1[iter]= -(p_m)
        # rew2[iter]= -(p_m)
    else:
        rew1= -1.0
        rew2= -1.0
            
            
    
    b_status.append(bat_level)
    pkt_miss.append(p_m)
    in_pkt.append(p_n)
    out_pkt.append(p_o)
    eff_in_pkt.append(p_i_ef)
    
    true_on.append(on)
    queue_loss.append(q_loss)
    pkt_entry.append(p_l)
    
    return l_state, l_state_sleep, rew1, rew2, q_f, p4j, pkt_in, [p_m, q_loss]














 


policy = 'RL'
num_runs = 100
performance = []



for run in range(num_runs):

    print('Run:',run)

    # nw_top = dict(N1=[-1,[],[],0,0.75],N2=[-1,[],[],0,0.15],N3=[['N1','N2'],[],[],0,-1],N4=[['N3'],[],[],1,-1])
    
    # nw_top = dict(N1=[-1,[],[],0,0.75],N2=[['N1'],[],[],0],N3=[['N2'],[],[],0],N4=[['N3'],[],[],0],N5=[['N4'],[],[],0])
    
    
    # nw_top = dict(N1=[-1,[],[],0,0.75],N2=[-1,[],[],0,0.25],N3=[-1,[],[],0,0.30],N4=[-1,[],[],0,0.50],N5=[-1,[],[],0,0.15])
    
    
    # nw_top = dict(N1=[-1,[],[],0,0.75],N2=[-1,[],[],0,0.75],N3=[-1,[],[],0,0.75],N4=[-1,[],[],0,0.75],N5=[-1,[],[],0,0.75],N6=[-1,[],[],0,0.75],N7=[-1,[],[],0,0.75],N8=[-1,[],[],0,0.75],N9=[-1,[],[],0,0.75],N10=[-1,[],[],0,0.75])
    
    # nw_top = dict(N1=[-1,[],[],0,0.25])
    
    nw_top = dict(N1=[-1,[],[],0,0.25],N2=[['N1'],[],[],0],N3=[-1,[],[],0,0.25],N4=[['N3'],[],[],0],N5=[-1,[],[],0,0.25],N6=[['N5'],[],[],0],N7=[-1,[],[],0,0.10],N8=[['N7'],[],[],0],N9=[-1,[],[],0,0.10],N10=[['N9'],[],[],0],N11=[-1,[],[],0,0.10],N12=[['N11'],[],[],0],N13=[-1,[],[],0,0.50],N14=[['N13'],[],[],0],N15=[-1,[],[],0,0.50],N16=[['N15'],[],[],0],N17=[-1,[],[],0,0.50],N18=[['N17'],[],[],0],N19=[-1,[],[],0,0.75],N20=[['N19'],[],[],0])
    
    
    # nw_top = dict(N19=[-1,[],[],0,0.75])
    
    
    
    
    
    
    """
    [[One-hop upstream nodes],[empty list for pkt gen for downstream nodes],[empty list for storing node's performance parameters],[upstream node index from which it is receiving packets],flow data rate if source]
    
    """
    
    
    
    for x_iter in nw_top:
        
            
        if nw_top[x_iter][0]!=-1:
            
            num_fl = len(nw_top[x_iter][0])
            pkt_src = []
            for fl_id in range(num_fl):
                pkt_src.append(nw_top[nw_top[x_iter][0][fl_id]][1][nw_top[x_iter][3]])
            source = False
        else:
            pkt_src = []
            pkt_rate =  nw_top[x_iter][4]
            num_fl = 1
            source = True
            
            
        queue_length = [[0] for _ in range(num_fl)]
        # pkt_miss = [[0] for _ in range(num_fl)]
        pkt_miss = [0]
        in_pkt = [0]
        out_pkt = [0]
        eff_in_pkt = [0]
        
        queue_loss = [0]
        pkt_miss_energy = [0]
        pkt_miss_intent = [0]
        
        pkt_entry = [0]
        
        true_on = [0]
        
        ns = 0.0
        q = [[0] for _ in range(num_fl)]
            
        
        
        state = 0
        state_sleep = 0
        pmr = [0.0, 0.0]
        
        
        action_s_list = []
        action_t_list = []
         
        
        s_pol = 0
        t_pol = 0
        
            
        p4j = [[] for _ in range(num_fl)]
    
    
        tmp_list = []    
        
        
        
        Q_table_s = np.zeros((num_states,num_action_s))
        Q_table_t = np.zeros((num_states,num_action_t))
        
        
        
        
        
        cur_st = 0
        b_status = [0]
        
        harv_enr = []
        
        q_src = 0
        
    
    
        for time in range(sim_time):
            
            
            p2t = []
            p2t_src = []
            
            for c in range(num_fl):
                p2t.append(np.zeros((1,history)))
            
            
            if bool(pkt_src):
                for c in range(num_fl):
                    p2t[c] = np.array([pkt_src[c][time*history:(time+1)*history]])
                    tmp_list.append(copy.deepcopy(p2t[c]))
            else:
                for c in range(num_fl):
                    p2t[c] = np.array([np.random.poisson(lam=pkt_rate, size=history).tolist()])
                    
                for x_it in p2t[c][0]:
                    if x_it == 0:
                        p2t_src.append(min(q_src,1))
                        q_src = max(q_src-1,0)
                    else:
                        p2t_src.append(min(x_it,1))
                    if x_it>1:
                        q_src += x_it-1
                for c in range(num_fl):
                    p2t[c] = copy.deepcopy(np.reshape(np.array(p2t_src),(1,history)))
        
            # action = np.array([random.randint(0,9)])
            
            epsilon = eps_max * math.exp(-(time / exp_decay))
            
            
            rf1 = random()
                
            if (rf1 < epsilon):
                act_s = randint(0, num_action_s-1)
            else:
                act_s = np.argmax(Q_table_s,1)[state_sleep]
                
            rf2 = random()
            
            
            if (rf2 < epsilon):
                act_t = randint(0, num_action_t-1)
            else:
                act_t = np.argmax(Q_table_t,1)[state]
            
            
            flow_sel = randint(0, num_fl-1)
        
                
            action_s_list.append(act_s/10) 
            action_t_list.append(act_t/10) 
            
            ns, ns_sleep, reward1, reward2, q, pd, p_in, new_pmr = network(np.array([act_s]), np.array([act_t]), time,q, pmr, p2t, policy,num_fl, flow_sel, source)
            
            
            
                
                
            Q_table_s[state_sleep][act_s]=Q_table_s[state_sleep][act_s]+alpha*(reward2+gamma*np.max(Q_table_s[ns_sleep])-Q_table_s[state_sleep][act_s])
            
            Q_table_t[state][act_t]=Q_table_t[state][act_t]+alpha*(reward1+gamma*np.max(Q_table_t[ns])-Q_table_t[state][act_t])
            
            
            state = copy.deepcopy(ns)
            state_sleep = copy.deepcopy(ns_sleep)
            pmr = copy.deepcopy(new_pmr)
                
                    
        nw_top[x_iter][1] = copy.deepcopy(p4j)
            
        
        miss_rate = [y*100/in_pkt[x] for x, y in enumerate(pkt_miss) if in_pkt[x]!=0]
        
        queue_loss_rate = [y*100/in_pkt[x] for x, y in enumerate(queue_loss) if in_pkt[x]!=0]
        
        miss_rate_2 = [y*100/pkt_entry[x] for x, y in enumerate(pkt_miss) if pkt_entry[x]!=0]
        
        
        qlen = []
        qlen_sd = []
        
        for x_it in range(num_fl):
            qlen.append(np.mean(queue_length[x_it][avg_strt:sim_time]))
            qlen_sd.append(np.std(queue_length[x_it][avg_strt:sim_time]))
        
        
        nw_top[x_iter][2] = copy.deepcopy([np.mean(miss_rate[avg_strt:sim_time]),qlen,np.mean(queue_loss_rate[avg_strt:sim_time]),np.mean(queue_loss[avg_strt:sim_time]), np.mean(action_s_list[avg_strt:sim_time]),np.mean(action_t_list[avg_strt:sim_time]),np.std(miss_rate[avg_strt:sim_time]),qlen_sd,np.std(queue_loss[avg_strt:sim_time]),np.mean(b_status[avg_strt:sim_time])])
    
    
    
    
    
        mvw = 100
        mvw_av = 3000
        
        
        # plt.figure()
        # plt.plot(moving_average(miss_rate,mvw),'b')
        # plt.plot(moving_average(miss_rate,mvw_av),'r')
        # plt.xlabel('Time duration (in frames)')
        # plt.ylabel('Average missed reception')
        # plt.figure()
        # plt.plot(moving_average(queue_length,mvw),'b')
        # plt.plot(moving_average(queue_length,mvw_av),'r')
        # plt.xlabel('Time duration (in frames)')
        # plt.ylabel('Average queue length')
        # plt.figure()
        # plt.plot(moving_average(queue_loss_rate,mvw),'b')
        # plt.plot(moving_average(queue_loss_rate,mvw_av),'r')
        # plt.xlabel('Time duration (in frames)')
        # plt.ylabel('Queue loss rate')
        # plt.figure()
        # plt.plot(moving_average(action_s_list,mvw),'b')
        # plt.plot(moving_average(action_s_list,mvw_av),'r')
        # plt.xlabel('Time duration (in frames)')
        # plt.ylabel('$P_{on}$')
        # plt.figure()
        # plt.plot(moving_average(action_t_list,mvw),'b')
        # plt.plot(moving_average(action_t_list,mvw_av),'r')
        # plt.xlabel('Time duration (in frames)')
        # plt.ylabel('$P_{tx}$')
    
    
    
    
    
    
    output_var = []
    
    for x in nw_top:
        output_var.append(nw_top[x][2])


    performance.append(output_var)



