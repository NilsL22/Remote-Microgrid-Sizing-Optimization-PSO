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
#it is necessary to define the number of threads for the Numba library. For different processors different number of threads might be optimal. For debugging however one needs to set the number of threads to 1. To change the number of threads one needs to restart the kernel.
num_threads = 1 
import os
os.environ["OMP_NUM_THREADS"] = str(num_threads)       # NumPy / MKL / OpenBLAS threads
os.environ["NUMBA_NUM_THREADS"] = str(num_threads)      # Numba threads

import numba
numba.set_num_threads(num_threads)                  # Explicit Numba thread setting


import numpy as np
from numpy.exceptions import AxisError
import pandas as pd
from matplotlib import pyplot as plt
import time
#from Schimpe_degradation_new_v5_daily import degr_semi_empirical_PSO
from Schimpe_degradation_numba_v2 import degr_semi_empirical_PSO
#from Schimpe_degradation_numba_v2 import degr_semi_empirical_PSO
from numba import njit
import datetime


def semi_emp_degr_future(Batt_degr_start, no_days, ch_throughput_start, q_throughput_start, no_years, loss_cyc_lt_start, loss_cyc_ht_start, loss_cal_start, Batt_cap_full_loc, make_plots):
        loss_cyc_lt_fut = loss_cyc_lt_start/np.sqrt(ch_throughput_start)*np.sqrt(ch_throughput_start/no_days*365*no_years) #based on the battery degradation model. More details are explained in the external file used to calculate degradation in calc_opex()
        loss_cyc_ht_fut = loss_cyc_ht_start/np.sqrt(q_throughput_start)*np.sqrt(q_throughput_start/no_days*365*no_years)
        loss_cyc_fut = loss_cyc_lt_fut + loss_cyc_ht_fut
        loss_cal_fut = loss_cal_start/np.sqrt(24*no_days)*np.sqrt(24*365*no_years)
        total_loss_fut = loss_cyc_fut + loss_cal_fut
        c_new_batt = Batt_cap_full_loc*Batt_cost_siz_energy
        deg_cost_fut = total_loss_fut/0.2*c_new_batt

        
        loss_cyc_over_years = np.zeros(365*no_years)
        loss_cal_over_years = np.zeros(365*no_years)
        for n1 in range(365*no_years):
            loss_cyc_over_years[n1] = loss_cyc_lt_start/np.sqrt(ch_throughput_start)*np.sqrt(ch_throughput_start/no_days*n1) + loss_cyc_ht_start/np.sqrt(q_throughput_start)*np.sqrt(q_throughput_start/no_days*n1)
            
        for n2 in range(365*no_years):
            loss_cal_over_years[n2] = loss_cal_start/np.sqrt(24*no_days)*np.sqrt(24*n2)
        
        total_loss_over_years = loss_cyc_over_years + loss_cal_over_years
        #deg_rate_fut = np.median(np.diff(total_loss_over_years,axis = 1),axis=1)
        #linearized_semi_emp  = np.arange(0,3650*deg_rate_fut,deg_rate_fut)
        
        
        if make_plots == "yes":
            
            #Distinction between calendar, cyclic and total degradation
            tick_labels = np.arange(0,12,2)
            plt.plot(np.arange(365*no_years), loss_cyc_over_years*100)
            plt.plot(np.arange(365*no_years), loss_cal_over_years*100)
            plt.plot(np.arange(365*no_years),total_loss_over_years*100)
            plt.xticks(np.arange(0,3650+365*2,365*2),tick_labels)
            plt.legend(["Cyclic Degradation", "Calendar Degradation","Total Degradation"], fontsize = 13)
            plt.ylim((0,20))
            plt.xlabel("Time [Years]", fontsize = 14)
            plt.ylabel("Degradation [%]", fontsize = 14)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.grid()
            plt.tight_layout()
            plt.show()
            
        return deg_cost_fut, loss_cyc_fut, loss_cal_fut, total_loss_fut, total_loss_over_years



@njit
def calc_degr(Batt_powers,Batt_cap_siz_max_SOC, Batt_cap_degr_loc): #energy-throughput model
    Batt_cost_siz_energy_loc = Batt_cost_siz_energy
    No_cycles_total = 4500 #full equivalent cycles
    SOH_final = 0.8
    SOH_initial = 1
    SOH_now = Batt_cap_degr_loc/Batt_cap_siz_max_SOC
    Usable_SOH = SOH_initial-SOH_final
    timestep = 1 # in hours
    E_life = No_cycles_total*Batt_cap_siz_max_SOC * 0.89 #Total energy throughput before SOH reaches 0.8. The multiplier 0.89 is added to to the maximum energy decrease over the lifetime which can be calculated from the area of a trapezoid
    E_delta = np.sum(np.abs(Batt_powers),axis = 1)*timestep
    Rel_deg = E_delta/E_life
    SOH_delta = Rel_deg*Usable_SOH
    SOH_new = SOH_now - SOH_delta
    Batt_cap_new = SOH_new*Batt_cap_siz_max_SOC
    c_new_batt = Batt_cap_siz_max_SOC*Batt_cost_siz_energy_loc #battery cost
    degr_cost = SOH_delta*c_new_batt
    return Batt_cap_new, degr_cost




def calc_capex(pos_siz_capex, allow_sw_degr_capex):
    # if switch degradation is not considered inverter cost is part of capex otherwise it is part of opex where it is considered as degradation cost
    # TODO this way of calculation inverter cost causes unfair comparison between the cases when inverter degradation cost is considered and when it is not
    if allow_sw_degr_capex == 1:
        Cost_array_capex = np.array([Batt_cost_siz_energy, PV_cost_siz_pan])
    else:
        Cost_array_capex = np.array([Batt_cost_siz_energy, PV_cost_siz_pan + PV_cost_siz_inv])
    capex = np.sum(Cost_array_capex[:pos_siz_capex.ndim]*pos_siz_capex,axis = 1) 
    
    return capex




@njit #Inside Numba functions it is more efficient to use for loops than array opeartions
def linear_interp(x, xp, fp): #Manual linear interpolation for Numba compatibility 
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
                    Load_array_opex, PV_max_array_opex, PV_cap_opex, i_EMS_iter, degr_type_opex, Ua_SOC_data_opex, total_ch_opex, total_q_opex,
                    total_days_opex, der_qch_opex, der_q_opex, loss_cal_opex, T_profile_opex, PV_curves_rel_opex,
                    Gen_cost_curve,no_of_days_opex, Gen_pow_max, allow_sw_degr_opex, Load_multiplier_202):
    eff_ch = 0.98 #Charging efficiency
    eff_dis = 0.98 #Discharging efficieny
    PV_cost_EMS = 0 #Operating cost of PV
    
    # match the shapes your degr_switch returns
    rel_switch_degr = np.zeros(n_part_EMS_opex,      dtype=np.float64)  
    T_switch_opex   = np.zeros((n_part_EMS_opex,24), dtype=np.float64)  
    degr_cost_switch= np.zeros(n_part_EMS_opex,      dtype=np.float64)
    opex_no_penalties = np.zeros(n_part_EMS_opex, dtype=np.float64)
    opex = np.zeros(n_part_EMS_opex, dtype=np.float64)
        
    
    Batt_cost_EMS = 0  # operation cost of the BESS system. Degradation cost is calculated separately
    # Adjust battery power based on charge/discharge efficiency
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances):
            if pos[p, t, 1] >= 0:
                pos[p, t, 1] *= eff_dis
            else:
                pos[p, t, 1] *= eff_ch

    # Fuel cost calculation based on a fuel efficiency curve
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
                opex_temp[p, t, k] = Cost_array[k] * abs(pos[p, t, k]) #Calculation of PV and battery operation costs
            opex_temp[p, t, 2] = fuel_cost[p, t] * pos[p, t, 2] #Calculation of fuel cost

    opex_time_step = np.sum(opex_temp, axis=1) # Sum of PV, battery and fuel costs for each particle at each time instance
    opex_particles = np.sum(opex_time_step, axis=1) # Sum of opex for each particle
    fuel_cost_array_opex = np.sum(opex_temp[:, :, 2], axis=1) #Sum of fuel cost for each particle

    # Battery SOC update
    SOC_curves[:, 0] = Last_SOC
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances):
            SOC_curves[p, t+1] = SOC_curves[p, t] - pos[p, t, 1]/Batt_cap_degr_loc2*100

    New_SOC = SOC_curves[:, -1] # SOC at the end of the day

    # SOC limit breach penalty
    SOC_limit_breach = np.zeros(n_part_EMS_opex)
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances+1):
            if SOC_curves[p, t] < 20 or SOC_curves[p, t] > 85: 
                SOC_limit_breach[p] += abs(SOC_curves[p, t]) #Penalize SOC values if they go over 85 or under 20
                # TODO adjust this way of penalizing since it penalizes breaches over 85 more than the ones under 20

    # Power balance violation
    power_balance_violation = np.zeros(n_part_EMS_opex)
    for p in range(n_part_EMS_opex): # Calculation of whether the total generated and stored power matches the total consumed power
        for t in range(no_time_instances):
            total_power = pos[p, t, 0] + pos[p, t, 1] + pos[p, t, 2]
            power_balance_violation[p] += abs(total_power - Load_array_opex[p, t])

    # Battery degradation
    Batt_powers = pos[:, :, 1].astype(np.float64)
    
    if degr_type_opex == 1: #semi-empirical battery degradation model
        Batt_cap_new, total_ch_new, total_q_new, loss_calendar_new, loss_cyclic_lt_new, loss_cyclic_ht_new, \
        derivative_q_ch_new, derivative_q_new, degr_cost_batt = degr_semi_empirical_PSO(
            Batt_cap_degr_loc2, Batt_powers, SOC_curves/100, Batt_cap_siz_max_opex, total_ch_opex,
            total_q_opex, total_days_opex, der_qch_opex, der_q_opex, loss_cal_opex, Ua_SOC_data_opex, no_of_days_opex
        )
    else: #Energy-throughput battery degradation model
        Batt_cap_new, degr_cost_batt = calc_degr(Batt_powers[:,:24],Batt_cap_siz_max_opex,Batt_cap_degr_loc2)
        #total_ch_new, total_q_new, loss_calendar_new, loss_cyclic_lt_new, loss_cyclic_ht_new, derivative_q_ch_new, derivative_q_new = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        total_ch_new        = np.zeros(n_part_EMS_opex, dtype=np.float64)
        total_q_new         = np.zeros(n_part_EMS_opex, dtype=np.float64)
        derivative_q_ch_new = np.zeros(n_part_EMS_opex, dtype=np.float64)
        derivative_q_new    = np.zeros(n_part_EMS_opex, dtype=np.float64)
        loss_calendar_new   = np.zeros(n_part_EMS_opex, dtype=np.float64)
        loss_cyclic_lt_new  = np.zeros(n_part_EMS_opex, dtype=np.float64)
        loss_cyclic_ht_new  = np.zeros(n_part_EMS_opex, dtype=np.float64)
    
    
    PV_pos = pos[:, :, 0].astype(np.float64)
    PV_max_array_degr = PV_max_array_opex[0, :] #maximum possible generated solar power with the given solar panel array size
    P_actual_rel = np.zeros((n_part_EMS_opex, no_time_instances))
    for p in range(n_part_EMS_opex):
        for t in range(no_time_instances):
            if PV_max_array_degr[t] > 0:
                P_actual_rel[p, t] = PV_pos[p, t] / PV_max_array_degr[t] #Relative PV power used in respect to the maximum possible

    Vin_sw = compute_v_index_fast(P_actual_rel, PV_curves_rel_opex) # Calculate the voltage of the solar panel array on the maximum power point curve
    rel_switch_degr, degr_cost_switch, T_switch_opex = degr_switch(PV_max_array_degr, PV_pos, Vin_sw, PV_cap_opex, T_profile_opex) #Calculate the switch degradation
    
    if allow_sw_degr_opex == 0: #if switch degradation is not allowed degradation cost is zero and inverter cost is considered in capex
        degr_cost_switch = np.zeros(n_part_EMS_opex)
        
        
    #Tuning of the penalty parameters
    Solar_growth = PV_cap_opex/5 #the reference of 5 kWp and 10kWh have been arbitrarily chosen
    Battery_energy_growth = Batt_cap_siz_max_opex/10
     
    #Depending on the size the penalties have to be adjusted sometime
    if Solar_growth >2.9:
        Solar_multiplier_pb = 0.87 
        Solar_multiplier_soc = 0.97 
    elif Solar_growth >2.25: 
        Solar_multiplier_pb = 0.85 
        Solar_multiplier_soc = 0.95 
    elif Solar_growth >1.8:
        Solar_multiplier_pb = 0.85 
        Solar_multiplier_soc = 1.0
    elif Solar_growth > 0.2: 
        Solar_multiplier_pb = 1.0 
        Solar_multiplier_soc = 1.0
    else:
        Solar_multiplier_pb = 10.0
        Solar_multiplier_soc = 10.0
  
    if Battery_energy_growth > 0.0:
        Battery_energy_multiplier = 1.0/Battery_energy_growth
    else:
        Battery_energy_multiplier = 100
    
    
    if allow_sw_degr_opex == 1:
        sw_pen = 1.15
    else:
        sw_pen = 1
    penalty_battery_limits_opex = 0.035*Solar_multiplier_soc*Battery_energy_multiplier*(Load_multiplier_202**2)*sw_pen
    penalty_power_balance_opex = 1.0*Solar_multiplier_pb*Battery_energy_multiplier*Load_multiplier_202*sw_pen
     
    
    opex_no_penalties = opex_particles + degr_cost_batt + degr_cost_switch
    opex = opex_particles + SOC_limit_breach * penalty_battery_limits_opex + power_balance_violation * penalty_power_balance_opex + degr_cost_batt + degr_cost_switch #opex calculation as a sum of the opeartional cost and the penalties

    return opex, New_SOC, Batt_cap_new, fuel_cost_array_opex, opex_no_penalties, \
           total_ch_new, total_q_new, derivative_q_ch_new, derivative_q_new, \
           loss_calendar_new, loss_cyclic_lt_new, loss_cyclic_ht_new, rel_switch_degr, T_switch_opex, degr_cost_switch




def dispatch_pso(sizing_limits,PV_pow_max_part,PV_cap_disp_loc, Ua_SOC_data_pso,degr_type,T_profile_pso, G_profile_pso, PV_LUT_pso, no_of_days_pso, Gen_cost_curve_pso, allow_sw_degr_pso):
    try:
        Batt_cap_max_dispatch = sizing_limits[0] #if the sizing is for more than one component
        PV_cap_dispatch = sizing_limits[1]
    except IndexError:
        Batt_cap_max_dispatch = sizing_limits #if sizing is only for the battery
        PV_cap_dispatch = PV_cap_disp_loc
    
    n_part_EMS = 1000 #due to the way the particles are initialized the number of particles should be a cube of an integer
    max_iter_EMS = 10
    c1_EMS = 2.6  # cognitive coefficient -> higher c1, higher exploration
    c2_EMS = 2 #social coefficient -> higher c2, higher exploitation   
    w_EMS = 0.8  # inertia -> higher w, higher exploration
    conv_threshold_EMS = 200 # number of iterations after which the optimization is stopped or velocity limits are changed
    
    
    Batt_pow_max_dispatch = Batt_cap_max_dispatch*pow_cap_ratio
    opex_over_time = np.zeros(no_of_days)
    opex_no_penalties_over_time = np.zeros(no_of_days)
    Batt_cap_degr = Batt_cap_max_dispatch
    Batt_cap_degr_over_time = np.zeros(no_of_days+1)
    Batt_cap_degr_over_time[0] = Batt_cap_degr
    SOC_over_days = np.zeros(no_of_days+1)
    SOC_over_days[0] = Batt_SOC
    pos_EMS_over_time = np.zeros((no_of_days, 24, 3))
    
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
    switch_degr_over_time = np.zeros((no_of_days+1))
    T_switch_over_time = np.zeros((no_of_days,24))
    degr_cost_sw_over_time = np.zeros(no_of_days)
    
    # to keep track of the opex for convergence threshold
    opex_check = np.zeros((no_of_days, max_iter_EMS))
    t2_ind = 0
    i_EMS_break = np.ones(no_of_days)*max_iter_EMS
    fuel_cost_over_time = np.zeros(no_of_days)
    
    Solar_growth = PV_cap_dispatch/5 #the reference of 5 kWp has been chosen by tuning for such a size; the vel limit is then adjusted proportionally to the size change
    Battery_power_growth = Batt_pow_max_dispatch/3.5
    Solar_vel_multiplier = np.min([Load_multiplier_20,Solar_growth])
    Gen_pow_max_pso = np.max(Load)
    for t2 in np.arange(0, no_of_days):  # for every day
        #Extract the PV curves for the weather of the corresponding day
        PV_curves_rel_pso = extract_PV_curves_from_LUT(T_profile_pso[t2,:], G_profile_pso[t2,:], PV_LUT_pso)
        # Create a load profile for each EMS particle
        Load_array = np.array([Load[t2, :] for t in range(n_part_EMS)])
        # Create a PV profile for each EMS particle
        PV_max_array = np.array([PV_pow_max_part[t2, :] for t in range(n_part_EMS)])
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

        pos_EMS[:, :, 1] = np.tile(np.linspace(-Batt_pow_max_dispatch, Batt_pow_max_dispatch, no_rep1).repeat(no_rep1), no_rep1)[:, np.newaxis]
        pos_EMS[:, :, 2] = np.linspace(0, Gen_pow_max, no_rep1).repeat(no_rep2)[:, np.newaxis]

        vel_EMS = np.zeros((n_part_EMS, no_time_instances, 3))

        max_bounds_EMS = np.array([[PV_pow_max_part[t2, t], Batt_pow_max_dispatch, Gen_pow_max] for t in range(no_time_instances)])
        min_bounds_EMS = [0, -Batt_pow_max_dispatch, 0]
        
        #Initialization of neighborhoods
        n_neigh = 35 #35
        neighborhoods = np.zeros((n_part_EMS,n_neigh)) #Initialize the neighborhood array
        neighborhoods[:,0] = np.arange(n_part_EMS) #The first column lists the particles in an ascending order
        neighborhoods[:,1:] = np.array([np.random.choice(n_part_EMS, size=n_neigh - 1, replace=False) for _ in range(n_part_EMS)]) #The particles indicated in the first column are assigned with a neighborhood of 19 other random particles
        neighborhoods = neighborhoods.astype(int)
        g_best_neigh_opex= np.ones(1000)*10**5 #Initial best opex of each neighborhood
        g_best_neigh_pos = np.zeros(pos_EMS.shape)
        flag = 0
        vel_max_EMS_array = np.array([0.125*Solar_vel_multiplier, 0.125*Battery_power_growth, 1*Load_multiplier_20])
        for i_EMS in range(max_iter_EMS):  # EMS PSO starts
            if i_EMS > conv_threshold_EMS and flag == 0: #after certain amount of iterations reduce the maximum velocity to limit the exploration space and allow for finding the solution more accurately
                vel_max_EMS_array = vel_max_EMS_array/2
                flag = 1

            opex, New_SOC_t2, Batt_cap_degr_array, fuel_cost_array, opex_no_penalties_pso, \
                total_ch_pso_array, total_q_pso_array, der_q_ch_array, der_q_array, \
                loss_cal_array, loss_cyclic_lt_array, loss_cyclic_ht_array, switch_degr_pso, T_switch_pso, degr_cost_sw_pso = calc_opex(pos_EMS, n_part_EMS, no_time_instances, 
                                                                                       Batt_cap_max_dispatch, Batt_cap_degr_over_time[t2_ind], SOC_over_days[t2_ind], 
                                                                                       Load_array, PV_max_array, PV_cap_dispatch, i_EMS, degr_type, Ua_SOC_data_pso, total_ch_over_time[t2_ind], 
                                                                                       total_q_over_time[t2_ind],t2+1, der_qch_over_time[t2_ind], der_q_over_time[t2_ind], loss_cal_over_time[t2_ind], T_profile_pso[t2,:], 
                                                                                       PV_curves_rel_pso, Gen_cost_curve_pso, no_of_days_pso, Gen_pow_max_pso, allow_sw_degr_pso, Load_multiplier_20)
    
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
            r1 = np.random.rand(n_part_EMS, no_time_instances, 3)
            r2 = np.random.rand(n_part_EMS, no_time_instances, 3)
            # Adjust the velocity of EMS
            vel_EMS = w_EMS*vel_EMS + c1_EMS*r1 * (p_best_pos_EMS - pos_EMS) + c2_EMS*r2*(g_best_neigh_pos - pos_EMS)
            vel_EMS = np.clip(vel_EMS, -vel_max_EMS_array, vel_max_EMS_array)
            # Update the position of the EMS particle
            pos_EMS = np.clip(pos_EMS + vel_EMS,min_bounds_EMS, max_bounds_EMS)
        
        opex_over_time[t2_ind], SOC_over_days[t2_ind+1], Batt_cap_degr_over_time[t2_ind+1], fuel_cost_over_time[t2_ind], \
            opex_no_penalties_over_time[t2_ind], total_ch_over_time[t2_ind+1], \
            total_q_over_time[t2_ind + 1], der_qch_over_time[t2_ind+1], \
            der_q_over_time[t2_ind+1], loss_cal_over_time[t2_ind+1],loss_cyc_lt_over_time[t2_ind+1], \
            loss_cyc_ht_over_time[t2_ind+1], switch_degr_over_time[t2_ind + 1], T_switch_over_time[t2_ind], degr_cost_sw_over_time[t2_ind] = calc_opex(np.array([g_best_neigh_pos[np.argmin(g_best_neigh_opex)]]), 1, no_time_instances, Batt_cap_max_dispatch, Batt_cap_degr_over_time[t2_ind], 
                                                                                                                       SOC_over_days[t2_ind], np.array([Load[t2_ind]]), PV_max_array, PV_cap_dispatch, i_EMS, degr_type, Ua_SOC_data_pso, total_ch_over_time[t2_ind], 
                                                                                                                       total_q_over_time[t2_ind] ,t2+1, der_qch_over_time[t2_ind], der_q_over_time[t2_ind] ,loss_cal_over_time[t2_ind], T_profile_pso[t2,:], 
                                                                                                                       PV_curves_rel_pso, Gen_cost_curve_pso, no_of_days_pso, Gen_pow_max_pso, allow_sw_degr_pso, Load_multiplier_20)
        pos_EMS_over_time[t2_ind] = g_best_neigh_pos[np.argmin(g_best_neigh_opex)]
        t2_ind += 1
    return opex_over_time, pos_EMS_over_time, Batt_cap_degr_over_time, opex_check, i_EMS_break, \
        opex_no_penalties_over_time, np.sum(fuel_cost_over_time), total_ch_over_time[-1], total_q_over_time[-1], \
        loss_cal_over_time[-1], np.sum(loss_cyc_lt_over_time), np.sum(loss_cyc_ht_over_time), switch_degr_over_time, T_switch_over_time , np.sum(degr_cost_sw_over_time)



def dispatch_RB(sizing_limits, PV_pow_max_RB, Load_rb_loc, degr_type, Ua_SOC_data_rb, T_profile_RB, G_profile_RB, PV_LUT_RB, allow_sw_degr, Gen_cost_curve_RB):
    # Based on the paper by Cicilio et al.
    try:
        Batt_cap_max_dispatch = sizing_limits[0]
    except IndexError:
        Batt_cap_max_dispatch = sizing_limits
    
   
    total_ch_over_time_rb = np.zeros(no_of_days+1)
    total_ch_over_time_rb[0] = 0
    total_q_over_time_rb = np.zeros(no_of_days+1)
    total_q_over_time_rb[0] = 0
    der_qch_over_time_rb = np.zeros(no_of_days+1)
    der_qch_over_time_rb[0] = 0
    der_q_over_time_rb = np.zeros(no_of_days+1)
    der_q_over_time_rb[0] = 0
    loss_cal_over_time_rb = np.zeros(no_of_days+1)
    loss_cal_over_time_rb[0] = 0
    loss_cyc_lt_over_time_rb = np.zeros(no_of_days+1)
    loss_cyc_lt_over_time_rb[0] = 0
    loss_cyc_ht_over_time_rb = np.zeros(no_of_days+1)
    loss_cyc_ht_over_time_rb[0] = 0
    
    
    Batt_max_pow_rb = Batt_cap_max_dispatch*pow_cap_ratio
    SOC = np.zeros((Load_rb_loc.shape[0],Load_rb_loc.shape[1]))
    SOC_limit_low = 20
    SOC_limit_high = 85
    P_batt = np.zeros(Load_rb_loc.shape)
    P_gen = np.zeros(Load_rb_loc.shape)
    P_dump = np.zeros(Load_rb_loc.shape)
    forecast_horizon = 12 #the original paper considers a horizon of 12 
    Batt_cap_rb = np.zeros(Load_rb_loc.shape[0]+1)
    Batt_cap_rb[0] = Batt_cap_max_dispatch
    degr_cost_tot_batt_RB = np.zeros(Load_rb_loc.shape[0])
    degr_cost_tot_sw_RB = np.zeros(Load_rb_loc.shape[0])
    rel_switch_degr_RB = np.zeros(Load_rb_loc.shape[0])
    T_switch_RB = np.zeros(Load_rb_loc.shape)
    
    
    Prev_SOC = Batt_SOC
    for t1 in range(Load_rb_loc.shape[0]):
        T_profile_day = T_profile_RB[t1,:]
        G_profile_day = G_profile_RB[t1,:]
        PV_curves_rel_RB = extract_PV_curves_from_LUT(T_profile_day, G_profile_day, PV_LUT_RB)
        for t2 in range(Load_rb_loc.shape[1]):
            if t1 == 0 and t2 ==0:
                Prev_SOC = Batt_SOC
            elif t1>0 and t2 == 0:
                Prev_SOC = SOC[t1-1,-1]
            else:
                Prev_SOC = SOC[t1, t2-1]
            E_batt_left = (Prev_SOC)/100*Batt_cap_rb[t1]
            #Calculate the remaining load till the end of the horizon or the end of the day
            #TODO This does not fully match with the paper and needs to be fixed as the horizon of 12 hours should always be taken. Since representative days are taken one probably needs to use a copy of the same day.
            E_load_left = np.sum(Load_rb_loc[t1,t2:min(t2+forecast_horizon, Load_rb_loc.shape[1]-t2)])
            P_batt_max_ch = min(Batt_max_pow_rb, (Batt_cap_rb[t1]*SOC_limit_high/100-E_batt_left)/time_step)
            P_batt_max_dis = min(Batt_max_pow_rb, ((Prev_SOC-SOC_limit_low)/100)*Batt_cap_rb[t1]/time_step)
            if PV_pow_max_RB[t1,t2] > 0:
                if PV_pow_max_RB[t1,t2] < Load_rb_loc[t1,t2]:    
                    if ((E_batt_left-SOC_limit_low/100*Batt_cap_rb[t1]) > E_load_left) and (P_batt_max_dis > Load_rb_loc[t1,t2]):
                        P_gen[t1,t2] = 0
                        P_batt[t1,t2] = Load_rb_loc[t1,t2] - PV_pow_max_RB[t1,t2]
                        P_dump[t1,t2] = 0
                    else:
                        P_gen[t1,t2] = min(Gen_pow_max, (Load[t1,t2] - PV_pow_max_RB[t1,t2] + P_batt_max_ch))
                        P_batt[t1,t2] = -(P_gen[t1,t2] - (Load[t1,t2] - PV_pow_max_RB[t1,t2]))
                        P_dump[t1,t2] = 0
                else:
                    P_gen[t1,t2] = 0
                    P_batt[t1,t2] = -min((PV_pow_max_RB[t1,t2] - Load_rb_loc[t1,t2]), P_batt_max_ch)
                    P_dump[t1,t2] = PV_pow_max_RB[t1,t2] + P_batt[t1,t2] - Load_rb_loc[t1,t2]
            else:
                if ((E_batt_left-SOC_limit_low/100*Batt_cap_rb[t1]) > E_load_left) and (P_batt_max_dis > Load_rb_loc[t1,t2]):
                    P_gen[t1,t2] = 0
                    P_batt[t1,t2] = Load_rb_loc[t1,t2]
                    P_dump[t1,t2] = 0
                else:
                    P_gen[t1,t2] = min(Gen_pow_max, Load_rb_loc[t1,t2] + P_batt_max_ch)
                    P_batt[t1,t2] = -(P_gen[t1,t2] - Load_rb_loc[t1,t2])
                    P_dump[t1,t2] = 0
            SOC[t1,t2] = Prev_SOC - P_batt[t1,t2]/(Batt_cap_rb[t1]/time_step)*100
        P_PV_dispatched = PV_pow_max_RB - P_dump
        if degr_type == 1: #semi-empirical battery degradation model
            Batt_cap_rb[t1+1], total_ch_new, total_q_new, loss_calendar_new, loss_cyclic_lt_new, loss_cyclic_ht_new, derivative_q_ch_new, derivative_q_new, degr_cost_tot_batt_RB[t1] = degr_semi_empirical_PSO(Batt_cap_rb[t1], np.array([P_batt[t1]]), np.array([SOC[t1]/100]) , Batt_cap_max_dispatch, total_ch_over_time_rb[t1], total_q_over_time_rb[t1], t1+1, der_qch_over_time_rb[t1], der_q_over_time_rb[t1], loss_cal_over_time_rb[t1], Ua_SOC_data_rb, no_of_days) 
            
            total_ch_over_time_rb[t1+1] = total_ch_new
            total_q_over_time_rb[t1+1] = total_q_new
            der_qch_over_time_rb[t1+1] = derivative_q_ch_new
            der_q_over_time_rb[t1+1] = derivative_q_new
            loss_cal_over_time_rb[t1+1] = loss_calendar_new
            loss_cyc_lt_over_time_rb[t1+1] = loss_cyclic_lt_new
            loss_cyc_ht_over_time_rb[t1+1] = loss_cyclic_ht_new
        else: #energy-throughput battery degradation model
            Batt_cap_rb[t1+1], degr_cost_tot_batt_RB[t1] = calc_degr(np.array([P_batt[t1]]), Batt_cap_max_dispatch, Batt_cap_rb[t1]) #P_batt is put inside another matrix to match with the form of the calc_degr function
        
        P_actual_rel = np.zeros((1,24))
        for m in range(24):
            if P_PV_dispatched[0,m] > 0:
                P_actual_rel[0,m] = P_PV_dispatched[t1,m] / PV_pow_max_RB[t1,m]
        Vin_sw = compute_v_index_fast(P_actual_rel, PV_curves_rel_RB)
        rel_switch_degr_RB[t1], degr_cost_tot_sw_RB[t1], T_switch_RB[t1,:] = degr_switch(PV_pow_max_RB[t1,:], P_actual_rel, Vin_sw, Batt_cap_max_dispatch, T_profile_day)
        
        if allow_sw_degr == 0:
            degr_cost_tot_sw_RB[t1] = 0
            
            
    fuel_rel_output = P_gen/np.max(Load_rb_loc)
    fuel_cost = np.interp(fuel_rel_output,Gen_cost_curve_RB[:,0],Gen_cost_curve_RB[:,1]) #the cost of the fuel depends on the operating point inside the generator efficiency curve
    opex = np.sum(fuel_cost*P_gen) + np.sum(degr_cost_tot_batt_RB) + np.sum(degr_cost_tot_sw_RB)
    power_flows = np.stack((P_PV_dispatched,P_batt,P_gen),axis = -1)
    fuel_cost_RB = np.sum(fuel_cost*P_gen)
    
    return opex, power_flows, SOC, Batt_cap_rb, P_dump, fuel_cost_RB, total_ch_over_time_rb[-1], total_q_over_time_rb[-1], loss_cal_over_time_rb[-1], np.sum(loss_cyc_lt_over_time_rb), np.sum(loss_cyc_ht_over_time_rb), rel_switch_degr_RB, T_switch_RB, np.sum(degr_cost_tot_sw_RB)


def calc_NPC(pos_siz_NPC, capex_NPC, opex_1y, Batt_cap_degr_NPC, fuel_cost_fraction_NPC, total_ch_NPC, total_q_NPC, loss_cal_NPC, loss_cyc_lt_NPC, loss_cyc_ht_NPC, degr_type, degr_cost_sw_NPC, no_pen): #need to adjust for the growth in load and degradation of solar panels
    try:
        batt_cap = pos_siz_NPC[0] #if the sizing is for more than one component
    except IndexError:
        batt_cap = pos_siz_NPC #if sizing is only for the battery
    discount_rate = 0.08
    inflation_rate = 1.02
    lifetime = 20 #years
    #Battery lifetime estimation
    total_loss_fut = np.zeros(lifetime+1)
    degr_cost_batt_NPC = np.zeros(lifetime)
    if degr_type == 0: #if simple
        Delta_SOH = 0.2
        SOH_1y = (1 - (Batt_cap_degr_NPC/batt_cap))/no_of_days*365 #SOH after 1 year #SOH after 1 year
        if SOH_1y == 0 or Delta_SOH/SOH_1y > 100:
            year_batt_repl = lifetime
        else:
            year_batt_repl = int(Delta_SOH/SOH_1y)
        for n0 in range(lifetime+1):
            total_loss_fut[n0] = SOH_1y*(n0+1)
            if n0>0:
                degr_cost_batt_NPC[n0-1] = (total_loss_fut[n0]-total_loss_fut[n0-1])*batt_cap*Batt_cost_siz_energy/0.2
    else:
        loss_cyc_lt_fut = 0
        loss_cyc_ht_fut = 0
        for n in range(lifetime+1):
            if total_ch_NPC > 0.0:
                loss_cyc_lt_fut = loss_cyc_lt_NPC/np.sqrt(total_ch_NPC)*np.sqrt(total_ch_NPC/no_of_days*365*n)
            if total_q_NPC > 0.0:
                loss_cyc_ht_fut = loss_cyc_ht_NPC/np.sqrt(total_q_NPC)*np.sqrt(total_q_NPC/no_of_days*365*n)
            loss_cyc_fut = loss_cyc_lt_fut + loss_cyc_ht_fut
            loss_cal_fut = loss_cal_NPC/np.sqrt(24*no_of_days)*np.sqrt(24*365*n)
            total_loss_fut[n] = loss_cyc_fut + loss_cal_fut
            if n>0:
                degr_cost_batt_NPC[n-1] = (total_loss_fut[n]-total_loss_fut[n-1])/0.2*batt_cap*Batt_cost_siz_energy
        SOH_fut = 1 - total_loss_fut
        year_batt_repl = np.argmin(SOH_fut[SOH_fut > 0.8] - 0.8)
    
    opex_1y = opex_1y/no_of_days*365
    degr_cost_sw_1y = degr_cost_sw_NPC/no_of_days*365
    opex_lifetime = np.zeros(lifetime)
    opex_lifetime[0] = opex_1y
    opex_fuel = opex_1y*fuel_cost_fraction_NPC # if fuel fraction is very low the cost goes to infinity
    if EMS_strategy == 'opt' and no_pen == 'no':
        for t in range(lifetime):
            opex_lifetime[t] = opex_fuel*inflation_rate**t/((1+discount_rate)**t)*(1+0.01*t)  + degr_cost_batt_NPC[t]/(1+discount_rate)**t + degr_cost_sw_1y/((1+discount_rate)**t)*(1+total_loss_fut[t]) + (opex_1y- opex_fuel - degr_cost_sw_1y - degr_cost_batt_NPC[0]*0.1)
    else:
        for t in range(lifetime):
            opex_lifetime[t] = opex_fuel*inflation_rate**t/((1+discount_rate)**t)*(1+0.01*t)  + degr_cost_batt_NPC[t]/(1+discount_rate)**t + degr_cost_sw_1y/((1+discount_rate)**t)*(1+total_loss_fut[t]) 
    NPC = capex_NPC+ np.sum(opex_lifetime)
    return NPC, year_batt_repl



def pso_sizing(EMS_strategy_pso, year_opt_pso, PV_cap, no_of_days_siz, degr_type_batt_siz, allow_sw_degr_siz, Ua_SOC_data_siz, PV_LUT_siz, G_profile_siz, T_profile_siz, Gen_cost_curve_siz):
    
    #PSO parameters
    max_iter_siz = 5
    c1_siz = 2
    c2_siz = 2
    w_siz = 1
    conv_threshold_siz = 25
    
    g_best_batt_repl_year = lifetime
    g_best_capex = 0
    g_best_fuel_fraction = 0
    NPC_check = np.zeros(max_iter_siz)
    i_break_siz = max_iter_siz
    g_best_cost_siz = 10**5
    #Initialization of the sizing PSO 
    if year_opt_pso == lifetime: #Optimization at year 20 -> get the cap of solar panels and battery
        n_part_siz = 49 # should be a square of an integer for an optimal initialization
        n_rep_siz = round(np.sqrt(n_part_siz))
        p_best_cost_siz = np.ones(n_part_siz)*10**5
        n_dim_siz = 2
        vel_siz = np.zeros((n_part_siz,n_dim_siz))
        vel_max_siz_array = np.array([2,2])
        p_best_pos_siz = np.zeros((n_part_siz,n_dim_siz))
        g_best_pos_siz = np.zeros(n_dim_siz)
        max_bounds_siz = [Batt_cap_max,PV_power_max_siz]
        min_bounds_siz = [0.01,0]
        pos_siz = np.zeros((n_part_siz,n_dim_siz))
        pos_siz[:,0] = np.tile(np.linspace(min_bounds_siz[0],Batt_cap_max,n_rep_siz),n_rep_siz)
        pos_siz[:,1] = np.linspace(0,PV_power_max_siz,n_rep_siz).repeat(n_rep_siz)
          
        
    g_best_cost_siz_over_time = np.zeros(max_iter_siz)
    g_best_pos_siz_over_time = np.zeros((max_iter_siz,n_dim_siz))
    
    #Sizing PSO starts
    for i_siz in range(max_iter_siz):
        if i_siz == 24:
            print("Last_iteration")
        print("Sizing iteration: ",i_siz)
        capex = calc_capex(pos_siz, allow_sw_degr_siz)
        if year_opt_pso == lifetime: #Calculate the corresponding pv output curve for the chosen solar capacity
            PV_pow_max = pos_siz[:,1][:,np.newaxis,np.newaxis]*pv_power
            
        for p_siz in range(n_part_siz):
            if EMS_strategy_pso == "opt": #optimization based EMS
                opex_over_time, pos_EMS_over_time, Batt_cap_degr_over_time, opex_check_siz, i_EMS_break_siz, \
                    opex_no_penalties_siz, fuel_cost_siz, total_ch_siz, total_q_siz, loss_cal_siz, \
                        loss_cyc_lt_siz, loss_cyc_ht_siz, switch_degr_over_time_siz, T_switch_over_time_siz, degr_cost_sw_siz= dispatch_pso(pos_siz[p_siz],PV_pow_max[p_siz],PV_cap, Ua_SOC_data, degr_type_batt_siz, T_profile_siz, G_profile_siz, PV_LUT_siz, no_of_days_siz, Gen_cost_curve, allow_sw_degr_siz)
                opex = np.sum(opex_over_time)
            else: #Rule-based EMS
                try:                           
                    pv_power_rb = pv_power*pos_siz[p_siz][1]
                except IndexError:
                    pv_power_rb = pv_power*PV_cap
                opex, pos_EMS_over_time, SOC_over_time, Batt_cap_degr_over_time, P_curt_siz, fuel_cost_siz, \
                    total_ch_siz, total_q_siz, loss_cal_siz, loss_cyc_lt_siz, loss_cyc_ht_siz, switch_degr_over_time_siz, T_switch_over_time_siz, degr_cost_sw_siz = dispatch_RB(pos_siz[p_siz],pv_power_rb,Load, degr_type_batt_siz, Ua_SOC_data,T_profile, G_profile, PV_LUT, allow_sw_degr_siz,Gen_cost_curve)
            #PSO based on the NPC
            fuel_cost_fraction_siz = fuel_cost_siz/opex
            NPC, battery_replacement_year = calc_NPC(pos_siz[p_siz],capex[p_siz],opex,Batt_cap_degr_over_time[-1], fuel_cost_fraction_siz, total_ch_siz, total_q_siz, loss_cal_siz, loss_cyc_lt_siz, loss_cyc_ht_siz, degr_type_batt_siz, degr_cost_sw_siz, 'no') # does not make sense to take outside the for loop since otherwise it is hard to keep track with Batt capacity degradation
            if NPC < g_best_cost_siz:    # If the total cost is less that the global best, update the global and personal best cost and save the position
                g_best_cost_siz = NPC
                p_best_cost_siz[p_siz] = NPC
                g_best_pos_siz = pos_siz[p_siz]
                p_best_pos_siz[p_siz] = pos_siz[p_siz]
                g_best_batt_repl_year = battery_replacement_year
                g_best_capex = capex[p_siz]
                g_best_fuel_fraction = fuel_cost_fraction_siz
            elif NPC < p_best_cost_siz[p_siz]:   #If the total cost is less than the particle's best update the personal best cost and save the position
                p_best_cost_siz[p_siz] = NPC
                p_best_pos_siz[p_siz] = pos_siz[p_siz] 
        #Calculation of the random parameters
        NPC_check[i_siz] = g_best_cost_siz
        if i_siz > conv_threshold_siz and NPC_check[i_siz - conv_threshold_siz] - NPC_check[i_siz] < 1:
            i_break_siz = i_siz
            break
        
        r3 = np.random.rand(*pos_siz.shape)
        r4 = np.random.rand(*pos_siz.shape)
        
        vel_siz = w_siz*vel_siz + c1_siz*r3*(p_best_pos_siz - pos_siz) + c2_siz*r4*(g_best_pos_siz - pos_siz)
        vel_siz = np.clip(vel_siz,-vel_max_siz_array,vel_max_siz_array) #keep velocity within defined bounds
        pos_siz = pos_siz + vel_siz
        pos_siz = np.clip(pos_siz,min_bounds_siz,max_bounds_siz) #keep position within defined limits
        g_best_cost_siz_over_time[i_siz] = g_best_cost_siz
        g_best_pos_siz_over_time[i_siz,:] = g_best_pos_siz
    
    return g_best_pos_siz, g_best_batt_repl_year,g_best_pos_siz_over_time, g_best_cost_siz, g_best_capex





########################################################################################

start = time.time()

#Solar Power - aquired from renewables.ninja in Benin 
no_of_days = 4
#pv_power_ready = np.load("pv_power_ready_new_2_morning.npy")
#pv_power = pv_power_ready[:no_of_days]
pv_power = np.load("PV_profile_40.npy")[:no_of_days,:]

Ua_SOC_data = pd.read_csv("Anode_voltage_vs_SOC_v2.csv").to_numpy()
PV_LUT = np.load("PV_LUT_3.npy")
G_profile = np.load("Irradiance_profile_40.npy")[:no_of_days,:]
T_profile_rounded = np.round(np.load("Temperature_profile_40.npy")*2)/2
T_profile = T_profile_rounded[:no_of_days,:]

#Fuel consumption curve
carbon_tax = 1
Diesel_price_lit = 0.7*1.2*carbon_tax
Fuel_eff_curve = np.load("Fuel_efficiency_curve_L_kWh.npy")
Gen_cost_curve = np.zeros((1000, 2))
Gen_cost_curve[:, 0] = np.linspace(0, 1, 1000)
# The generator should not operate below 30% of the nominal load to avoid damage
Gen_cost_curve[:300, 1] = 100
Gen_cost_curve[300:, 1] = Fuel_eff_curve[300:]*Diesel_price_lit
Gen_cost_curve[0] = 0


#Load - aquired by using RAMP
lifetime = 20
Load_ready = np.load("repr_days_load_morning.npy") 
Load_growth = 1.05 #per year
Load_multiplier_20 = Load_growth**lifetime
#Load = Load_ready[:no_of_days,:]/1000*Load_multiplier_20 #Year 20 -> change back to basic Load and implement the growth in the sizing func
Load = np.zeros((no_of_days,24))
Load[:int(no_of_days/2),:] = Load_ready[:int(no_of_days/2),:]
Load[int(no_of_days/2):no_of_days,: ] = Load_ready[20:20+int(no_of_days/2),:] 
Load = Load/1000*Load_multiplier_20
Gen_pow_max = np.max(Load) #kW
time_step = 1 #in hours
no_time_instances = 24

PV_cost_siz_pan = 400 #per kWp solar panels + inverter if inverter cost of 150 per kWp is added solar becomes more expensive than batter
PV_cost_siz_inv = 500
PV_cost_EMS = 0 
Batt_cost_siz_energy = 500 #per kWh from solartopstore; 
Batt_cost_siz_power = 0 #per kW
Batt_cost_EMS = 0 #in principle 0 unless some kind of maintanence is included or the additional energy of the AC/ degr cost is calculated separately
#Battery 
Batt_SOC = 40 #%
eff_ch = 0.98
eff_dis = 0.98
Batt_cap_max = 40
pow_cap_ratio = 0.35
PV_power_max_siz = 25




#Comment out the corresponding one
#EMS_strategy = "RB" #Rule-based EMS
EMS_strategy = "opt" #optimisation-based EMS

#degr_type_batt_final =0 #simple
degr_type_batt_final = 1 #"semi_emp"
#allow_sw_degr_final = 0 #"no"
allow_sw_degr_final = 1 # "yes"
year_opt = lifetime

if EMS_strategy == "RB":
    from Switch_degr_func_RB import extract_PV_curves_from_LUT, degr_switch, compute_v_index_fast
else:
    from Switch_degr_func_v3 import extract_PV_curves_from_LUT, degr_switch, compute_v_index_fast

Final_sys_size_20, year_batt_repl_1_v1,best_sizing_over_time_20, Best_NPC_20, Best_capex_20  = pso_sizing(EMS_strategy,year_opt,PV_power_max_siz, no_of_days, degr_type_batt_final, allow_sw_degr_final, Ua_SOC_data, PV_LUT, G_profile, T_profile, Gen_cost_curve)
#Final_sys_size_20 = np.array([29.5,6.5])
#year_batt_repl_1_v1 = 12

# =============================================================================
# #Load adjustment depending on the replacement year of the battery
# Load_multiplier_20 = Load_growth**year_batt_repl_1_v1
# Load = Load_ready[:no_of_days,:]/1000*Load_multiplier_20 #Year 20 -> change back to basic Load and implement the growth in the sizing func
# Gen_pow_max = np.max(Load) #kW
# print("Gen_pow_max_repl:", Gen_pow_max)
# 
# Final_sys_size_repl1, year_batt_repl_1, best_sizing_over_time_repl, Best_NPC_repl, Best_capex_20_repl = pso_sizing(EMS_strategy,year_batt_repl_1_v1,Final_sys_size_20[1],no_of_days, degr_type_batt_final, allow_sw_degr_final, Ua_SOC_data, PV_LUT, G_profile, T_profile, Gen_cost_curve)
# 
# =============================================================================


end = time.time()
Time_taken = end-start
print("Time taken:", Time_taken)

##############################################################################################

#Solve the optimal power flow again for the optimal sizing

 
PV_pow_max_final = Final_sys_size_20[1]*pv_power
Ua_SOC_data = pd.read_csv("Anode_voltage_vs_SOC_v2.csv").to_numpy()

if EMS_strategy == "opt":
    time_new = time.perf_counter()
    opex_over_time_final, pos_EMS_over_time_final, Batt_cap_degr_over_time_final, opex_check_final, \
        i_EMS_break_final, opex_no_penalties_final, fuel_cost_final, total_ch_final, total_q_final, \
            loss_cal_final, loss_cyc_lt_final, loss_cyc_ht_final, \
                switch_degr_over_time_final, T_switch_over_time_final, degr_cost_sw_final  = dispatch_pso(Final_sys_size_20, PV_pow_max_final,Final_sys_size_20[1], Ua_SOC_data, degr_type_batt_final,T_profile, G_profile, PV_LUT, no_of_days, Gen_cost_curve, allow_sw_degr_final) 
    time_new2 = time.perf_counter()
    opex_no_penalties_total_final = np.sum(opex_no_penalties_final)
    opex_final = np.sum(opex_over_time_final)
    fuel_fraction_final = np.sum(fuel_cost_final)/np.sum(opex_over_time_final)
    Batt_power_final = pos_EMS_over_time_final[:,:,1].reshape(24*no_of_days)
    SOC_over_time_final = np.zeros(len(Batt_power_final)+1)
    SOC_over_time_final[0] = Batt_SOC
    for n in range(1,len(SOC_over_time_final)):
        SOC_over_time_final[n] = SOC_over_time_final[n-1] - Batt_power_final[n-1]/Final_sys_size_20[0]*100 
    Power_balance_final = np.sum(pos_EMS_over_time_final,axis = 2)-Load
    Rel_power_balance_error = np.sum(abs(Power_balance_final))/np.sum(Load)
    #Best_NPC_20_no_pen, year_batt_repl_final_NPC = calc_NPC(Final_sys_size_20, Best_capex_20, opex_no_penalties_total_final, Batt_cap_degr_over_time_final[-1], 1, total_ch_final, total_q_final, loss_cal_final, loss_cyc_lt_final, loss_cyc_ht_final, degr_type_batt_final,degr_cost_sw_final,'yes')
    P_curtailed_final = np.where(PV_pow_max_final - pos_EMS_over_time_final[:,:,0] > 0, PV_pow_max_final - pos_EMS_over_time_final[:,:,0], 0)
    P_curtailed_final_total = np.sum(P_curtailed_final)
    print("New time:", time_new2-time_new)
    print("OPEX: ", opex_no_penalties_total_final/no_of_days)
    #print("NPC:", Best_NPC_20_no_pen )
else: #RB dispatch
    pv_power_rb_final = pv_power*Final_sys_size_20[1]
    dispatch_time_RB_start = time.perf_counter()
    opex_final, pos_EMS_over_time_final, SOC_over_time_final, Batt_cap_degr_over_time_final, P_curt_siz_final, fuel_cost_siz_final, total_ch_final, total_q_final, loss_cal_final, loss_cyc_lt_final, loss_cyc_ht_final,switch_degr_over_time_final, T_switch_over_time_final, degr_cost_sw_final = dispatch_RB(Final_sys_size_20,pv_power_rb_final,Load, degr_type_batt_final, Ua_SOC_data, T_profile, G_profile, PV_LUT, allow_sw_degr_final,Gen_cost_curve)
    dispatch_time_RB_end = time.perf_counter()
    print("RB Dispatch Time: ", dispatch_time_RB_end - dispatch_time_RB_start)
    Batt_power_final = pos_EMS_over_time_final[:,:,1].reshape(24*no_of_days)
    SOC_over_time_final = np.zeros(len(Batt_power_final)+1)
    SOC_over_time_final[0] = Batt_SOC
    for n in range(1,len(SOC_over_time_final)):
        SOC_over_time_final[n] = SOC_over_time_final[n-1] - Batt_power_final[n-1]/Final_sys_size_20[0]*100 
    P_curtailed_final_total = np.sum(P_curt_siz_final)
    
    Best_capex_20 = calc_capex(Final_sys_size_20,1)
    Best_capex_20 = np.sum(Best_capex_20)
    #Best_NPC_20, year_batt_repl_final_NPC = calc_NPC(Final_sys_size_20, Best_capex_20, opex_final, Batt_cap_degr_over_time_final[-1], 1, total_ch_final, total_q_final, loss_cal_final, loss_cyc_lt_final, loss_cyc_ht_final, degr_type_batt_final,degr_cost_sw_final)
    print("OPEX: ", opex_final/no_of_days)
    #print("NPC: ", Best_NPC_20)
no_of_years = 10
if degr_type_batt_final == 1: # "semi_emp":
    plots_option = "yes"
    deg_cost_2, loss_cyc_2, loss_cal_2, total_loss_2, total_loss_over_years_final = semi_emp_degr_future(Batt_cap_degr_over_time_final[-1], no_of_days, total_ch_final, total_q_final, no_of_years, loss_cyc_lt_final, loss_cyc_ht_final, loss_cal_final, Final_sys_size_20[0], plots_option)
else:
    total_loss_2 = (1 - Batt_cap_degr_over_time_final[-1]/Batt_cap_degr_over_time_final[0])/no_of_days*3650

    
#print("CAPEX: ", Best_capex_20)

#print("NPC: ", Best_NPC_20)
#print("NPC: ", Best_NPC_20_no_pen)
print("PV capacity: ", Final_sys_size_20[1])
print("Battery capacity: ", Final_sys_size_20[0])
print("Battery throughput per day: ", np.sum(abs(pos_EMS_over_time_final[:,:,1])/no_of_days))
print("Generator energy per day: ", np.sum(pos_EMS_over_time_final[:,:,2])/no_of_days)
print("Total Battery Degradation: ", total_loss_2)
print("Curtailed Power per day: ", P_curtailed_final_total/no_of_days)
print("Total switch degradation: ", np.sum(switch_degr_over_time_final)/no_of_days*3650)
print("Fraction of energy generated by fuel:", np.sum(pos_EMS_over_time_final[:,:,2])/(np.sum(pos_EMS_over_time_final[:,:,0])+np.sum(pos_EMS_over_time_final[:,:,2])))

#Plot the dispatch
#Mean dispatch

#mean_PV_power = np.mean(pos_EMS_over_time_final[:,:,0], axis = 0)
#mean_batt_power = np.mean(pos_EMS_over_time_final[:,:,1], axis = 0)
#mean_gen_power = np.mean(pos_EMS_over_time_final[:,:,2], axis = 0)

# =============================================================================
# time_axis = [datetime.datetime(2024, 1, 1, 8) + datetime.timedelta(hours=i) for i in range(24)]  # 08:00 to 08:00 next day
# tick_times = [time_axis[0] + datetime.timedelta(hours=i) for i in range(0, 24, 2)]
# fig, ax = plt.subplots(figsize = (15,6))
# 
# plt.plot(time_axis, mean_PV_power)
# plt.plot(time_axis, mean_batt_power)
# plt.plot(time_axis, mean_gen_power)
# 
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)   
# ax.set_xticklabels([dt.strftime('%H:%M') for dt in tick_times], rotation=45)
# plt.legend(["Dispatched PV power","Dispatched Battery Power", "Dispatched Generator Power"], fontsize = 13)
# plt.xlabel("Time", fontsize = 16)
# plt.ylabel("Dispatched Power, kW", fontsize = 16)
# plt.show()
# =============================================================================

