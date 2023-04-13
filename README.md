# Interactive RL for Harvesting

File ID "Harvesting_RL_Policy.py" is for simulating RL-based transmission-sleep scheduling

File ID "Harvesting_Hybrid_Policy.py" is for simulating Hybrid sleep scheduling used for comparison

File ID "Harvesting_Naive_Policy_1_node.py" is for simulating naive transmission-sleep scheduling policy for benchmarking for a single node


Define the network topolgy in variable nw_top with format [[One-hop upstream nodes],[empty list for pkt gen for downstream nodes],[empty list for storing node's performance parameters],[upstream node index from which it is receiving packets],flow data rate if source]

Define number of runs and simulation time per run in variables num_runs and sim_time respectively

The performace output can be observed in list variable output_var
