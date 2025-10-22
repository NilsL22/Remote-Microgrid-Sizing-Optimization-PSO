# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 08:36:05 2025

@author: nilsl
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 14:47:13 2025

@author: nilsl
"""


#Move the PV curve extraction outside calc_opex


num_threads = 1
import os
os.environ["OMP_NUM_THREADS"] = str(num_threads)       # NumPy / MKL / OpenBLAS threads
os.environ["NUMBA_NUM_THREADS"] = str(num_threads)      # Numba threads

import numba
numba.set_num_threads(num_threads)                  # Explicit Numba thread setting

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
#from Schimpe_degradation_new_v5_daily import degr_semi_empirical_PSO
from Schimpe_degradation_numba_v2 import degr_semi_empirical_PSO
from Switch_degr_func_v3 import extract_PV_curves_from_LUT, degr_switch, compute_v_index_fast
from numba import njit

import random
#random.seed(14)  

def describe_arg(arg, name):
    import numpy as np
    if isinstance(arg, np.ndarray):
        print(f"{name}: ndarray, dtype={arg.dtype}, ndim={arg.ndim}, shape={arg.shape}, "
              f"strides={arg.strides}, C_contig={arg.flags['C_CONTIGUOUS']}, F_contig={arg.flags['F_CONTIGUOUS']}")
    else:
        print(f"{name}: {type(arg)}, value={arg}, np_type={np.array(arg).dtype}")

def semi_emp_degr_future(Batt_degr_start, no_days, ch_throughput_start, q_throughput_start, no_years, loss_cyc_lt_start, loss_cyc_ht_start, loss_cal_start, Batt_cap_full_loc, make_plots):
        Batt_cost_siz_energy_loc = 500
        loss_cyc_lt_fut = loss_cyc_lt_start/np.sqrt(ch_throughput_start)*np.sqrt(ch_throughput_start/no_days*365*no_years)
        loss_cyc_ht_fut = loss_cyc_ht_start/np.sqrt(q_throughput_start)*np.sqrt(q_throughput_start/no_days*365*no_years)
        loss_cyc_fut = loss_cyc_lt_fut + loss_cyc_ht_fut
        loss_cal_fut = loss_cal_start/np.sqrt(24*no_days)*np.sqrt(24*365*no_years)
        total_loss_fut = loss_cyc_fut + loss_cal_fut
        c_new_batt = Batt_cap_full_loc*Batt_cost_siz_energy_loc
        deg_cost_fut = total_loss_fut/0.2*c_new_batt

        
        loss_cyc_over_years = np.zeros(365*no_years)
        loss_cal_over_years = np.zeros(365*no_years)
        for n1 in range(365*no_years):
            loss_cyc_over_years[n1] = loss_cyc_lt_start/np.sqrt(ch_throughput_start)*np.sqrt(ch_throughput_start/no_days*n1) + loss_cyc_ht_start/np.sqrt(q_throughput_start)*np.sqrt(q_throughput_start/no_days*n1)
            
        for n2 in range(365*no_years):
            loss_cal_over_years[n2] = loss_cal_start/np.sqrt(24*no_days)*np.sqrt(24*n2)
        
        total_loss_over_years = loss_cyc_over_years + loss_cal_over_years
        
        
        
        if make_plots == "yes":
            tick_labels = np.arange(1,11,2)
            plt.plot(np.arange(365*no_years), loss_cyc_over_years)
            plt.plot(np.arange(365*no_years), loss_cal_over_years)
            plt.xticks(np.arange(0,3650,365*2),tick_labels)
            plt.legend(["Cyclic Degradation", "Calendar Degradation"])
            plt.ylim((0,0.2))
            plt.title("Degradation by its type")
            plt.xlabel("Time, Years", fontsize = 13)
            plt.ylabel("Degradation, %", fontsize = 13)
            plt.show()
            
            plt.plot(np.arange(365*no_years),total_loss_over_years)
            plt.xticks(np.arange(0,3650,365*2),tick_labels)
            plt.title("Total Degradation")
            
            simple_deg_day_3000 = 0.006879/100
            simple_deg_over_time_3000 = np.arange(0,simple_deg_day_3000*365*no_years,simple_deg_day_3000)
            plt.plot(np.arange(365*no_years),simple_deg_over_time_3000)
            
            simple_deg_day_4500 = 0.004586/100
            simple_deg_over_time_4500 = np.arange(0,simple_deg_day_4500*365*no_years,simple_deg_day_4500)
            plt.plot(np.arange(365*no_years+1),simple_deg_over_time_4500)
            
            simple_deg_day_6000 = 0.00344/100
            simple_deg_over_time_6000 = np.arange(0,simple_deg_day_6000*365*no_years,simple_deg_day_6000)
            plt.plot(np.arange(365*no_years),simple_deg_over_time_6000)
            
            plt.legend(["Semi-empirical Model", "Energy-Throughput Model 3000", "Energy-Throughput Model 4500","Energy-Throughpu Model 6000"], fontsize = 13)
            plt.xlabel("Time, Years", fontsize = 13)
            plt.ylabel("Degradation, %", fontsize = 13)
            plt.ylim((0,0.3))
            plt.show()
            
        return deg_cost_fut, loss_cyc_fut, loss_cal_fut, total_loss_fut, total_loss_over_years
    
@njit
def calc_degr(Batt_powers,Batt_cap_siz_max_SOC, Batt_cap_degr_loc): #energy-throughput model
    Batt_cost_siz_energy_loc = 500
    No_cycles_total = 4500 #full equivalent cycles
    SOH_final = 0.8
    SOH_initial = 1
    SOH_now = Batt_cap_degr_loc/Batt_cap_siz_max_SOC
    Usable_SOH = SOH_initial-SOH_final
    timestep = 1 # in hours
    E_life = No_cycles_total*2*Batt_cap_siz_max_SOC * 0.89 #Total energy throughput before SOH reaches 0.8. The multiplier 0.89 is added to to the maximum energy decrease over the lifetime which can be calculated from the area of a trapezoid
    E_delta = np.sum(np.abs(Batt_powers),axis = 1)*timestep
    Rel_deg = E_delta/E_life
    SOH_delta = Rel_deg*Usable_SOH
    SOH_new = SOH_now - SOH_delta
    #SOH_new = np.ones(n_part_EMS)
    Batt_cap_new = SOH_new*Batt_cap_siz_max_SOC
    RUL = (SOH_new - SOH_final)*No_cycles_total/Usable_SOH #Remaining useful lifetime
    c_new_batt = Batt_cap_siz_max_SOC*Batt_cost_siz_energy_loc
    #degr_rate = Usable_SOH/RUL
    #degr_cost = degr_rate*(c_new_batt - c_salv_batt) #accumulative cost -> cannot use for each cycle separately
    degr_cost = SOH_delta*c_new_batt
    return Batt_cap_new, degr_cost





def warmup_calc_opex(T_profile_warm_up):
    
    # Minimal sizes for quick compilation
    n_part_EMS_opex = 1000
    no_time_instances = 24
    
    # Dummy input arrays
    pos = np.zeros((n_part_EMS_opex, no_time_instances, 3), dtype=np.float64)
    Load_array_opex = np.zeros((n_part_EMS_opex, no_time_instances), dtype=np.float64)
    PV_max_array_opex = np.zeros((n_part_EMS_opex, no_time_instances), dtype=np.float64)
    Ua_SOC_data_opex = pd.read_csv("Anode_voltage_vs_SOC_v2.csv").to_numpy()
    PV_curves_rel_opex = np.load("PV_curves_rel_init.npy")

  
    Last_SOC = np.float64(50.0)
    Batt_cap_siz_max_opex = np.float64(100.0)
    Batt_cap_degr_loc2 = np.float64(100.0)
    i_EMS_iter = 0
    total_q_opex=np.float64(0.0)
    total_ch_opex=np.float64(0.0)
    total_days_opex=np.int64(1)
    der_qch_opex=np.float64(0.0)
    der_q_opex=np.float64(0.0)
    loss_cal_opex=np.float64(0.0)
    PV_size = np.float64(1.0)
    
    penalty_battery_limits_w = 0.035*Solar_multiplier_soc*Battery_energy_multiplier*Load_multiplier**2*2
    penalty_power_balance_w = 1*Solar_multiplier_pb*Battery_energy_multiplier*Load_multiplier
    
    calc_opex(
        pos, n_part_EMS_opex, no_time_instances, Batt_cap_siz_max_opex, Batt_cap_degr_loc2, Last_SOC,
        Load_array_opex, PV_max_array_opex, i_EMS_iter, Ua_SOC_data_opex, total_ch_opex, total_q_opex,
        total_days_opex, der_qch_opex, der_q_opex, loss_cal_opex, PV_size, T_profile_warm_up, PV_curves_rel_opex,
        Gen_cost_curve,
        penalty_battery_limits_w, penalty_power_balance_w, no_of_days,3.2
    )

    print("Warm-up complete! Numba JIT compilation done.")





@njit
def linear_interp(x, xp, fp):
    """Manual linear interpolation for Numba compatibility"""
    n = len(xp)
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        xi = x[i]
        if xi <= xp[0]:
            y[i] = fp[0]
        elif xi >= xp[-1]:
            y[i] = fp[-1]
        else:
            for j in range(n-1):
                if xp[j] <= xi <= xp[j+1]:
                    y[i] = fp[j] + (fp[j+1] - fp[j]) * (xi - xp[j]) / (xp[j+1] - xp[j])
                    break
    return y



@njit
def calc_opex(pos, n_part_EMS_opex, no_time_instances, Batt_cap_siz_max_opex, Batt_cap_degr_loc2, Last_SOC,
                    Load_array_opex, PV_max_array_opex, i_EMS_iter, Ua_SOC_data_opex, total_ch_opex, total_q_opex,
                    total_days_opex, der_qch_opex, der_q_opex, loss_cal_opex, PV_size, T_profile_opex, PV_curves_rel_opex,
                    Gen_cost_curve,
                    penalty_battery_limits_opex, penalty_power_balance_opex, no_of_days_opex, Gen_pow_max):
    eff_ch = 0.98
    eff_dis = 0.98 
    PV_cost_EMS = 0
    Batt_cost_EMS = 0  # in principle 0 unless some kind of maintanence is included or the additional energy of the AC/ degr cost is calculated separately
    
    
    # Adjust battery power based on charge/discharge efficiency
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances):
            if pos[p, t, 1] >= 0:
                pos[p, t, 1] *= eff_dis
            else:
                pos[p, t, 1] *= eff_ch

    # Fuel cost calculation
    fuel_rel_output = pos[:, :, 2] / Gen_pow_max
    fuel_cost = linear_interp(fuel_rel_output.flatten(), Gen_cost_curve[:, 0], Gen_cost_curve[:, 1])
    fuel_cost = fuel_cost.reshape((n_part_EMS_opex, no_time_instances))

    # Cost array
    SOC_curves = np.zeros((n_part_EMS_opex, no_time_instances + 1))
    Cost_array = np.array([PV_cost_EMS, Batt_cost_EMS])
    opex_temp = np.zeros(pos.shape)
    
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances):
            for k in range(2):
                opex_temp[p, t, k] = Cost_array[k] * abs(pos[p, t, k])
            opex_temp[p, t, 2] = fuel_cost[p, t] * pos[p, t, 2]

    opex_time_step = np.sum(opex_temp, axis=1)
    opex_particles = np.sum(opex_time_step, axis=1)
    fuel_cost_array_opex = np.sum(opex_temp[:, :, 2], axis=1)

    # Battery SOC update
    SOC_curves[:, 0] = Last_SOC
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances):
            SOC_curves[p, t+1] = SOC_curves[p, t] - pos[p, t, 1]/Batt_cap_degr_loc2*100

    New_SOC = SOC_curves[:, -1]

    # SOC limit breach penalty
    SOC_limit_breach = np.zeros(n_part_EMS_opex)
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances+1):
            if SOC_curves[p, t] < 20 or SOC_curves[p, t] > 85:
                SOC_limit_breach[p] += abs(SOC_curves[p, t])

    # Power balance violation
    power_balance_violation = np.zeros(n_part_EMS_opex)
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances):
            total_power = pos[p, t, 0] + pos[p, t, 1] + pos[p, t, 2]
            power_balance_violation[p] += abs(total_power - Load_array_opex[p, t])

    # Battery degradation
    Batt_powers = pos[:, :, 1].astype(np.float64)
    if degr_type_batt == "semi_emp":
        Batt_cap_new, total_ch_new, total_q_new, loss_calendar_new, loss_cyclic_lt_new, loss_cyclic_ht_new, \
        derivative_q_ch_new, derivative_q_new, degr_cost_batt = degr_semi_empirical_PSO(
            Batt_cap_degr_loc2, Batt_powers, SOC_curves/100, Batt_cap_siz_max_opex, total_ch_opex,
            total_q_opex, total_days_opex, der_qch_opex, der_q_opex, loss_cal_opex, Ua_SOC_data_opex, no_of_days_opex
        )
    else:
        Batt_cap_new, degr_cost_batt = calc_degr(Batt_powers[:,:24],Batt_cap_siz_max_opex,Batt_cap_degr_loc2)
        total_ch_new, total_q_new, loss_calendar_new, loss_cyclic_lt_new, loss_cyclic_ht_new, derivative_q_ch_new, derivative_q_new = np.array([0,0,0,0,0,0,0])
    
    #Switch degradation
    
    PV_pos = pos[:, :, 0].astype(np.float64)
    PV_max_array_degr = PV_max_array_opex[0, :]
    P_actual_rel = np.zeros((n_part_EMS_opex, no_time_instances))
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances):
            if PV_max_array_degr[t] > 0:
                P_actual_rel[p, t] = PV_pos[p, t] / PV_max_array_degr[t]

    Vin_sw = compute_v_index_fast(P_actual_rel, PV_curves_rel_opex)
    rel_switch_degr, degr_cost_switch, T_switch_opex = degr_switch(PV_max_array_degr, PV_pos, Vin_sw, PV_size, T_profile_opex)
   
    
    if allow_sw_degr == "no":
        degr_cost_switch = 0 #For testing the impact of switch degradation
        #rel_switch_degr = np.zeros(n_part_EMS_opex)
        #T_switch_opex = np.zeros((n_part_EMS_opex,24))
    
    opex_no_penalties = opex_particles + degr_cost_batt + degr_cost_switch
    opex = opex_particles + SOC_limit_breach * penalty_battery_limits_opex + power_balance_violation * penalty_power_balance_opex + degr_cost_batt + degr_cost_switch

    return opex, New_SOC, Batt_cap_new, fuel_cost_array_opex, opex_no_penalties, \
           total_ch_new, total_q_new, derivative_q_ch_new, derivative_q_new, \
           loss_calendar_new, loss_cyclic_lt_new, loss_cyclic_ht_new, rel_switch_degr, T_switch_opex

def opex_pso(sizing_limits, PV_pow_max_part, Ua_SOC_data_pso, T_profile_pso, G_profile_pso, PV_LUT_pso, no_of_days_pso, Gen_cost_curve_pso):
    
    if allow_sw_degr == "yes":
        sw_pen = 1.15
    else:
        sw_pen = 1
    
    penalty_battery_limits_pso = 0.035*Solar_multiplier_soc*Battery_energy_multiplier*(Load_multiplier**2)*sw_pen #*2
    penalty_power_balance_pso = 1*Solar_multiplier_pb*Battery_energy_multiplier*(Load_multiplier**1)*sw_pen #*1
    Gen_pow_max_opex = np.max(Load)
    
    opex_over_time = np.zeros(no_of_days)
    opex_no_penalties_over_time = np.zeros(no_of_days)
    Batt_cap_degr = sizing_limits[2]
    Batt_cap_degr_over_time = np.zeros(no_of_days+1)
    Batt_cap_degr_over_time[0] = Batt_cap_degr
    total_ch_over_time = np.zeros(no_of_days+1)
    total_ch_over_time[0] = 0
    total_q_over_time = np.zeros(no_of_days+1)
    total_q_over_time[0] = 0
    der_qch_over_time = np.zeros(no_of_days+1)
    der_qch_over_time[0] = 0
    der_q_over_time = np.zeros(no_of_days+1)
    der_q_over_time[0] = 0
    loss_cal_over_time = np.zeros(no_of_days+1)
    loss_cal_over_time[0] = 0
    loss_cyc_lt_over_time = np.zeros(no_of_days+1)
    loss_cyc_lt_over_time[0] = 0
    loss_cyc_ht_over_time = np.zeros(no_of_days+1)
    loss_cyc_ht_over_time[0] = 0
    SOC_over_days = np.zeros(no_of_days+1)
    SOC_over_days[0] = Batt_SOC
    switch_degr_over_time = np.zeros((no_of_days+1))
    T_switch_over_time = np.zeros((no_of_days,24))
    pos_EMS_over_time = np.zeros((no_of_days, 24, 3))
    opex_check = np.zeros((no_of_days, max_iter_EMS)) # to keep track of the opex for convergence threshold
    t2_ind = 0
    i_EMS_break = np.ones(no_of_days)*max_iter_EMS
    fuel_cost_over_time = np.zeros(no_of_days)
    for t2 in np.arange(0, no_of_days):  # for every day
        PV_curves_rel_pso = extract_PV_curves_from_LUT(T_profile_pso[t2,:], G_profile_pso[t2,:], PV_LUT_pso)
        
        # Create a load profile for each EMS particle
        Load_array = np.array([Load[t2, :] for t in range(n_part_EMS)]).astype(np.float64)
        # Create a PV profile for each EMS particle
        PV_max_array = np.array([PV_pow_max_part[t2, :]
                                for t in range(n_part_EMS)]).astype(np.float64)
        p_best_opex_EMS = np.ones(n_part_EMS)*10**8
        p_best_pos_EMS = np.zeros((n_part_EMS, no_time_instances, 3))
        pos_EMS = np.zeros((n_part_EMS, no_time_instances, 3))
        
        # Initializing EMS particles
        # number of repeated elements for initialization -> n_part_EMS should be dividable by this number
        no_rep1 = round(n_part_EMS**(1/3))
        no_rep2 = round(n_part_EMS**(2/3))

        # Grid-spread initialization of pos_EMS
        for t in range(no_time_instances):  # max PV depends on the time moment
            pos_EMS[:, t, 0] = np.tile(np.linspace(0, PV_pow_max_part[t2, t], no_rep1), no_rep2)

        pos_EMS[:, :, 1] = np.tile(np.linspace(-sizing_limits[1], sizing_limits[1], no_rep1).repeat(no_rep1), no_rep1)[:, np.newaxis]
        pos_EMS[:, :, 2] = np.linspace(0, Gen_pow_max, no_rep1).repeat(no_rep2)[:, np.newaxis]

        vel_EMS = np.zeros((n_part_EMS, no_time_instances, 3))

        max_bounds_EMS = np.array([[PV_pow_max_part[t2, t], sizing_limits[1], Gen_pow_max] for t in range(no_time_instances)])
        min_bounds_EMS = [0, -sizing_limits[1], 0]
        
        #Initialization of neighborhoods
        n_neigh = 35 #35
        neighborhoods = np.zeros((n_part_EMS,n_neigh))
        neighborhoods[:,0] = np.arange(n_part_EMS)
        neighborhoods[:,1:] = np.array([np.random.choice(n_part_EMS, size=n_neigh - 1, replace=False) for _ in range(n_part_EMS)])
        neighborhoods = neighborhoods.astype(int)
        g_best_neigh_opex= np.ones(1000)*10**5
        g_best_neigh_pos = np.zeros(pos_EMS.shape)
        flag = 0
        
        vel_max_EMS_array = np.array([0.15*Solar_vel_multiplier, 0.15*Battery_power_growth, 1*Load_multiplier])
        
        start1 = time.time()
        r1 = np.empty((n_part_EMS, no_time_instances, 3))
        r2 = np.empty((n_part_EMS, no_time_instances, 3))
        vel_threshold = 150
        for i_EMS in range(max_iter_EMS):  # EMS PSO starts
            #start2 = time.perf_counter()
            if i_EMS > vel_threshold and flag == 0:
                vel_max_EMS_array = vel_max_EMS_array/2
                flag = 1
            elif i_EMS > vel_threshold*2.5 and flag == 1:
                vel_max_EMS_array[0:2] = vel_max_EMS_array[0:2]/3
                flag = 2
        
         
            opex, New_SOC_t2, Batt_cap_degr_array, fuel_cost_array, opex_no_penalties_pso, total_ch_pso_array, total_q_pso_array, \
                der_q_ch_array, der_q_array, loss_cal_array, loss_cyclic_lt_array, loss_cyclic_ht_array, switch_degr, \
                    T_switch_pso = calc_opex(pos_EMS, n_part_EMS, no_time_instances, sizing_limits[2], Batt_cap_degr_over_time[t2_ind], 
                                             SOC_over_days[t2_ind], Load_array, PV_max_array, i_EMS, Ua_SOC_data_pso, total_ch_over_time[t2_ind], 
                                             total_q_over_time[t2_ind],t2+1, der_qch_over_time[t2_ind], der_q_over_time[t2_ind], loss_cal_over_time[t2_ind], sizing_limits[0], 
                                             T_profile_pso[t2,:], PV_curves_rel_pso, Gen_cost_curve_pso, penalty_battery_limits_pso,penalty_power_balance_pso,no_of_days_pso, Gen_pow_max_opex)
            
            opex_neigh = opex[neighborhoods]
            best_neigh_ind = np.argmin(opex_neigh,axis = 1) # the index of the minimum opex of each neighborhood
            best_neigh_opex = opex_neigh[np.arange(n_part_EMS),best_neigh_ind]
            best_neigh_part_ind = neighborhoods[np.arange(n_part_EMS),best_neigh_ind] #the corresponding particle indices to the minimum value of each neighborhood
            best_neigh_part_pos = pos_EMS[best_neigh_part_ind]
            
            #determine the global best of each neighborhood
            new_mask_neigh = best_neigh_opex < g_best_neigh_opex 
            g_best_neigh_opex[new_mask_neigh] = best_neigh_opex[new_mask_neigh]
            g_best_neigh_pos[new_mask_neigh] = best_neigh_part_pos[new_mask_neigh]
            
            #determine the p_best of each particle
            new_mask_part = opex < p_best_opex_EMS 
            p_best_opex_EMS[new_mask_part] = opex[new_mask_part]
            p_best_pos_EMS[new_mask_part] = pos_EMS[new_mask_part]
            
            #Checking for early convergence
            opex_check[t2_ind, i_EMS] = np.min(g_best_neigh_opex)
            if i_EMS > conv_threshold_EMS and opex_check[t2_ind, i_EMS - conv_threshold_EMS] - opex_check[t2_ind, i_EMS] < 0.05:
                i_EMS_break[t2_ind] = i_EMS
                break
            
            # Calculation of the random parameters
            r1[:,:,:] = np.random.rand(n_part_EMS, no_time_instances, 3)
            r2[:,:,:] = np.random.rand(n_part_EMS, no_time_instances, 3)
            # Adjust the velocity of EMS
            vel_EMS = w_EMS*vel_EMS + c1_EMS*r1 * (p_best_pos_EMS - pos_EMS) + c2_EMS*r2*(g_best_neigh_pos - pos_EMS)
            vel_EMS = np.clip(vel_EMS, -vel_max_EMS_array, vel_max_EMS_array)
            # Update the position of the EMS particle
            pos_EMS = np.clip(pos_EMS + vel_EMS,min_bounds_EMS, max_bounds_EMS)
            #end2 = time.perf_counter()
            #print("Single iteration:", end2-start2)
        end1 = time.time()
        print("PSO single day:", end1-start1)
        opex_over_time[t2_ind], SOC_over_days[t2_ind+1], Batt_cap_degr_over_time[t2_ind+1], fuel_cost_over_time[t2_ind], \
            opex_no_penalties_over_time[t2_ind], total_ch_over_time[t2_ind+1], total_q_over_time[t2_ind + 1], \
            der_qch_over_time[t2_ind+1], der_q_over_time[t2_ind+1], loss_cal_over_time[t2_ind+1],loss_cyc_lt_over_time[t2_ind+1], \
                loss_cyc_ht_over_time[t2_ind+1], switch_degr_over_time[t2_ind+1], T_switch_over_time[t2_ind] = calc_opex(np.array([g_best_neigh_pos[np.argmin(g_best_neigh_opex)]]), 1, no_time_instances, 
                                                                                                                         sizing_limits[2], Batt_cap_degr_over_time[t2_ind], SOC_over_days[t2_ind], np.array([Load[t2_ind]]).astype(np.float64), 
                                                                                                                         PV_max_array, i_EMS, Ua_SOC_data_pso, total_ch_over_time[t2_ind], total_q_over_time[t2_ind] ,t2+1, der_qch_over_time[t2_ind], 
                                                                                                                         der_q_over_time[t2_ind] ,loss_cal_over_time[t2_ind], sizing_limits[0], T_profile_pso[t2,:], PV_curves_rel_pso,Gen_cost_curve_pso, penalty_battery_limits_pso,
                                                                                                                         penalty_power_balance_pso,no_of_days_pso, Gen_pow_max_opex)
        pos_EMS_over_time[t2_ind] = g_best_neigh_pos[np.argmin(g_best_neigh_opex)]
        t2_ind += 1
    return opex_over_time, pos_EMS_over_time, Batt_cap_degr_over_time, opex_check, i_EMS_break, fuel_cost_over_time, opex_no_penalties_over_time, loss_cal_over_time,loss_cyc_lt_over_time, loss_cyc_ht_over_time, total_ch_over_time,total_q_over_time, switch_degr_over_time, T_switch_over_time

no_of_days = 40
Ua_SOC_data = pd.read_csv("Anode_voltage_vs_SOC_v2.csv").to_numpy()
#pv_data = pd.read_csv("ninja_pv_temp_irr.csv")
PV_LUT = np.load("PV_LUT_3.npy")
G_profile = np.load("Irradiance_profile_40.npy")[:no_of_days,:]
T_profile_rounded = np.round(np.load("Temperature_profile_40.npy")*2)/2
T_profile = T_profile_rounded[:no_of_days,:]
pv_power = np.load("PV_profile_40.npy")[:no_of_days,:]


# Solar Power - aquired from renewables.ninja in Benin

# =============================================================================
# pv_power_ready = np.load("pv_power_ready_new_2_morning.npy")
# #pv_power_ready = np.load("pv_power_ready_new_2_morning_365d.npy") #for testing battery models
# pv_power = pv_power_ready[:no_of_days]
# =============================================================================

# Load - aquired by using RAMP
Load_ready = np.load("repr_days_load_morning.npy")
#Load_ready = np.load("Load_365_dry_rainy_v2.npy") #for testing battery models -> not shifted
Load_multiplier = 1.05**20
Load = Load_ready[:no_of_days, :]/1000*Load_multiplier  # convert from W to kW.
Load = Load.reshape((no_of_days, 24))
time_step = 1  # in hours
no_time_instances = 24




# Costs
# with a multiplier of 0.4 capex takes over
# in Benin based on some quick googling (EUR/liter) -> could increase due to transport (multiplier) -> calculate the distance between the closest gas station and the village
Diesel_price_lit = 0.7*1.2
Fuel_eff_curve = np.load("Fuel_efficiency_curve_L_kWh.npy")
Gen_cost_curve = np.zeros((1000, 2))
Gen_cost_curve[:, 0] = np.linspace(0, 1, 1000)
# The generator should not operate below 30% of the nominal load to avoid damage
Gen_cost_curve[:300, 1] = 100
Gen_cost_curve[300:, 1] = Fuel_eff_curve[300:]*Diesel_price_lit
Gen_cost_curve[0] = 0
Gen_pow_max = np.max(Load)  # kW
PV_cost_EMS = 0

Batt_cost_EMS = 0  # in principle 0 unless some kind of maintanence is included or the additional energy of the AC/ degr cost is calculated separately
# Battery
Batt_SOC = 40  # %



g_best_pos_siz = np.array([6.5, 33*0.35, 33])  # [5, 3.5, 10] 
batt_pow_cap_ratio = g_best_pos_siz[1]/g_best_pos_siz[2] #only considered between 0.1 and 0.5
PV_pow_max_final = g_best_pos_siz[0]*pv_power



Solar_growth = g_best_pos_siz[0]/5 #the reference of 5 kWp has been chosen by tuning for such a size; the vel limit is then adjusted proportionally to the size change
Battery_power_growth = g_best_pos_siz[1]/3.5
Battery_energy_growth = g_best_pos_siz[2]/10
Solar_vel_multiplier = np.min([Load_multiplier,Solar_growth])



if Solar_growth >2.9:
    Solar_multiplier_pb = 0.87 #0.82
    Solar_multiplier_soc = 0.97 #0.95
elif Solar_growth >2.25: 
    Solar_multiplier_pb = 0.85 #0.8
    Solar_multiplier_soc = 0.95 #0.9
elif Solar_growth >1.8:
    Solar_multiplier_pb = 0.85 #0.82
    Solar_multiplier_soc = 1
elif Solar_growth > 0.2: 
    Solar_multiplier_pb = 1 
    Solar_multiplier_soc = 1
else:
    Solar_multiplier_pb = 10
    Solar_multiplier_soc = 10




Battery_energy_multiplier = 1/Battery_energy_growth



# EMS PSO hyperparameters -> needs to be inside the sizing for reinitialization which depends on the size
n_part_EMS = 1000 #for good initialization it should be a cube of an integer
max_iter_EMS = 1000
conv_threshold_EMS = 300  # custom threshold



c1_EMS = 2.6  # 2.6 -> cognitive -> higher c1, higher exploration 2.1 #2.5,1.5,0.6 works for solar cap of 4kWp for load factor of 1.05 and battery capacity of 30kWh
c2_EMS = 2 # 2 -> social -> higher c2, higher exploitation   
w_EMS = 0.8  # 0.8 -> inertia -> higher w, higher exploration
#allow_sw_degr = "no"
allow_sw_degr = "yes"
#degr_type_batt = "simple"
degr_type_batt = "semi_emp"


#############
start_init = time.perf_counter()
warmup_calc_opex(T_profile[0,:].astype(np.float64))
end_init = time.perf_counter()
print("Initialization time:", end_init-start_init)
# Solve the optimal power flow again for the optimal sizing


# Optimization-based EMS
start = time.time()
opex_over_time_final, pos_EMS_over_time_final, Batt_degr_over_time_final, opex_check_final, i_EMS_break_final, fuel_cost_final, opex_no_penalties_over_time_final, loss_cal_over_time_final, loss_cyc_lt_over_time_final, loss_cyc_ht_over_time_final, total_ch_over_time_final, total_q_over_time_final, switch_degr_over_time_final, T_switch_over_time_final= opex_pso(g_best_pos_siz, PV_pow_max_final,Ua_SOC_data, T_profile, G_profile, PV_LUT, no_of_days, Gen_cost_curve)
end = time.time()
print("Time taken:", end-start)

opex_total_final = np.sum(opex_over_time_final)
opex_no_pen_total_final = np.sum(opex_no_penalties_over_time_final)
fuel_fraction_cost_final = fuel_cost_final/np.sum(opex_over_time_final)
Batt_power_final = pos_EMS_over_time_final[:, :, 1]
SOC_over_time_final = np.zeros(Batt_power_final.shape)
for n1 in range(SOC_over_time_final.shape[0]):
    for n2 in range(SOC_over_time_final.shape[1]):
        if n1 == 0 and n2 == 0:
            SOC_over_time_final[n1, n2] = Batt_SOC - \
                Batt_power_final[n1, n2]/Batt_degr_over_time_final[n1]*100
        elif n2 == 0:
            SOC_over_time_final[n1, n2] = SOC_over_time_final[n1-1, -1] - \
                Batt_power_final[n1, n2]/Batt_degr_over_time_final[n1]*100
        else:
            SOC_over_time_final[n1, n2] = SOC_over_time_final[n1, n2-1] - \
                Batt_power_final[n1, n2]/Batt_degr_over_time_final[n1]*100

Power_balance_final = np.sum(pos_EMS_over_time_final, axis=2)-Load
Rel_power_balance_error = np.sum(abs(Power_balance_final))/np.sum(Load)

P_curtailed_final = np.where(PV_pow_max_final - pos_EMS_over_time_final[:,:,0] > 0, PV_pow_max_final - pos_EMS_over_time_final[:,:,0], 0)
P_curtailed_over_time = np.sum(P_curtailed_final, axis = 1)
P_curtailed_final_total = np.sum(P_curtailed_final)

# Calculating future degradation
no_of_years = 10


print("Relative Error: ", Rel_power_balance_error)
print("Opex no pen per day:", opex_no_pen_total_final/no_of_days)
print("Battery Throughput per day", np.sum(abs(pos_EMS_over_time_final[:,:,1]))/no_of_days)
print("Generator Throughput per day:", np.sum(pos_EMS_over_time_final[:,:,2])/no_of_days)

if degr_type_batt == "simple":
    total_degr_10y = (1-Batt_degr_over_time_final[-1]/Batt_degr_over_time_final[0])/no_of_days*3650
else:
    deg_cost_2, loss_cyc_2, loss_cal_2, total_degr_10y, total_loss_over_years_final = semi_emp_degr_future(Batt_degr_over_time_final[-1], no_of_days, total_ch_over_time_final[-1], total_q_over_time_final[-1], no_of_years, np.sum(loss_cyc_lt_over_time_final), np.sum(loss_cyc_ht_over_time_final), loss_cal_over_time_final[-1], g_best_pos_siz[2], "yes")
    
print("Total degradation battery in 10 years: ", total_degr_10y)
print("Total degradation switch in 10 years: ", np.sum(switch_degr_over_time_final)/no_of_days*3650)
print("Curtailed power per day: ", P_curtailed_final_total/no_of_days)
print("Mean Temperature: ", np.mean(T_switch_over_time_final))