# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:30:29 2025

@author: nilsl
"""

def degr_semi_empirical_PSO(Batt_cap_degr_old, batt_powers, batt_SOC , Batt_cap_full, total_ch_old, total_q_old, total_days, last_der_q_ch, last_der_q, last_loss_calendar,Ua_SOC_data, no_of_days_deg):
    import numpy as np
    
    
    #batt_SOC = batt_SOC[:,1:]
    
    
    temp = 23+273.15                   # Temperature in Kelvin
    Batt_Voltage = 51.2 #V

    # CALENDAR AGING
    R = 8.314
    t_ref = 298.15  # reference temperature
    k_cal_ref = 3.694*10**(-4)
    Ea = 20592  # activation energy
    alpha = 0.384
    k0 = 0.142
    F = 96485  # faraday constant
    ua_ref = 0.123  # reference potential
    
    
# =============================================================================
#     SOC_ind_matrix = np.zeros(batt_SOC.shape)
#     diffs = np.abs(batt_SOC[:, :, np.newaxis]*100 - Ua_SOC_data[:,0][np.newaxis, np.newaxis, :])
#     SOC_ind_matrix = np.argmin(diffs, axis=2)
#     SOC_ind_matrix = SOC_ind_matrix.astype(int)
# =============================================================================

#Faster - the Ua_SOC_data needs to have equaly distributed points
    SOC_lookup = Ua_SOC_data[:, 0]  # Must be sorted ascending
    target_SOC = batt_SOC * 100
    
    SOC_ind_matrix = np.clip(
        np.searchsorted(SOC_lookup, target_SOC),
        0, len(SOC_lookup) - 1
    ) 



    
    Ua = Ua_SOC_data[SOC_ind_matrix,1]
    f_SOC_cal = k0 + np.exp(alpha*F/R*(ua_ref - Ua)/t_ref)
    f_T_cal = np.exp(-Ea/R*(1/temp-1/t_ref))
    k_cal = k_cal_ref*f_T_cal*f_SOC_cal

    indices = np.tile(np.arange(1,batt_SOC.shape[1]+1),(batt_SOC.shape[0],1))
    derivative_time = (k_cal * 0.5) / np.sqrt((total_days-1)*24 + indices) 

    #loss_calendar = np.zeros(batt_SOC.shape)
    #derivative_time_copy = np.cumsum(derivative_time, axis = 1)
    loss_calendar = last_loss_calendar + np.sum(derivative_time, axis = 1)
    



    # CYCLIC AGEING CALCULATION BY ZERO CROSSOVER METHOD
    
    #i_ref = 3
    #cell_capacity_ah = 3  # cell capacity in Ah
    #c_rate_ref = i_ref/cell_capacity_ah
    
    
    # low temperature
    k_cyc_lt_ref = 4.009*10**(-4)
    Ea_cyc_lt = 55546
    
    #high temperature
    k_cyc_ht_ref = 1.456*10**(-4)
    Ea_cyc_ht = 32699
    #beta_lt = 2.64
    
    
    
    
    Batt_power_cyclic = batt_powers
    
    Batt_profile_current_cyclic = Batt_power_cyclic / Batt_Voltage * 1000 *(5/Batt_cap_full)
    #cell_current_cyclic = Batt_profile_current_cyclic / (25000/51.2/3) #25kWh battery -> c-rate has a significant influence on the degradation
    cell_current_cyclic = Batt_profile_current_cyclic * 3 / 70 #more approapriate for our battery
    
    cell_current_ch_cyclic = np.where(cell_current_cyclic < 0 ,abs(cell_current_cyclic),0)
    cell_throughput_total_ch = np.sum(abs(cell_current_ch_cyclic), axis = 1)
    cell_throughput_total = np.sum(abs(cell_current_cyclic), axis = 1)
    
    total_throughput_q_ch = cell_throughput_total_ch + total_ch_old #
    total_throughput_q = cell_throughput_total + total_q_old
    # Compute degradation rate coefficient (vectorized)
    #k_cyc_lt = (k_cyc_lt_ref * np.exp((Ea_cyc_lt / R) * ((1 / temp) - (1 / t_ref))) * np.exp(beta_lt * ( c_rate_cyclic-c_rate_ref)))
    k_cyc_lt = k_cyc_lt_ref * np.exp((Ea_cyc_lt / R) * ((1 / temp) - (1 / t_ref)))
    k_cyc_ht = k_cyc_ht_ref * np.exp((-Ea_cyc_ht / R) * (1/temp - 1/t_ref))
    
    sqrt_tt = np.where(total_throughput_q > 0.0, np.sqrt(total_throughput_q),0.0)
    sqrt_tt_ch = np.where(total_throughput_q_ch > 0.0, np.sqrt(total_throughput_q_ch),0.0)
    
    safe_q_mask = sqrt_tt > 0.0
    safe_qch_mask = sqrt_tt_ch > 0.0
    
    derivative_q = np.zeros_like(sqrt_tt)
    derivative_q[safe_q_mask] = k_cyc_ht/(2*sqrt_tt[safe_q_mask])
    
    derivative_q_ch = np.zeros_like(sqrt_tt_ch)
    derivative_q_ch[safe_qch_mask] = k_cyc_lt/(2*sqrt_tt_ch[safe_qch_mask])

    if total_days == 1:
        loss_cyclic_ht = k_cyc_ht_ref * sqrt_tt
        loss_cyclic_lt = k_cyc_lt_ref * sqrt_tt_ch
    else:
        loss_cyclic_ht = last_der_q * cell_throughput_total
        loss_cyclic_lt = last_der_q_ch * cell_throughput_total_ch
        
    total_loss_cyclic = loss_cyclic_lt + loss_cyclic_ht
    
    degr_old = 1 - Batt_cap_degr_old/Batt_cap_full
    loss_total = (loss_calendar-last_loss_calendar + total_loss_cyclic)    
    degr_new = degr_old + loss_total
    Batt_cap_degr_new = Batt_cap_full * (1 - degr_new)
    
    
    Batt_cost_siz_energy = 500
    c_new_batt = Batt_cap_full*Batt_cost_siz_energy

    #deg_cost_final = loss_total/0.2*c_new_batt
    
    #Degradation Cost
    no_years = 10
    Usable_SOH = 0.2
    
    
    loss_cyc_fut_10 = np.zeros((batt_powers.shape[0],no_years))
    loss_cal_fut_10 = np.zeros((batt_powers.shape[0],no_years))
    cyc_fut_ht = np.zeros((batt_powers.shape[0]))
    cyc_fut_lt = np.zeros((batt_powers.shape[0]))
    safe_mask_total_q = total_throughput_q > 0
    safe_mask_total_qch = total_throughput_q_ch > 0
    
    
    for n1 in range(no_years):
        cyc_fut_ht[safe_mask_total_q] = loss_cyclic_ht[safe_mask_total_q]/np.sqrt(total_throughput_q[safe_mask_total_q])*np.sqrt(total_throughput_q[safe_mask_total_q]/total_days*365*n1)
        cyc_fut_lt[safe_mask_total_qch] = loss_cyclic_lt[safe_mask_total_qch]/np.sqrt(total_throughput_q_ch[safe_mask_total_qch])*np.sqrt(total_throughput_q_ch[safe_mask_total_qch]/total_days*365*n1)
        cyc_fut = cyc_fut_ht + cyc_fut_lt
        if cyc_fut.size: #in some cases the cyc_fut array can be empty
            loss_cyc_fut_10[:,n1] = cyc_fut
        else: 
            loss_cyc_fut_10[:,n1] = np.zeros(loss_cyc_fut_10[:,n1].shape)
        loss_cal_fut_10[:,n1] = loss_calendar/np.sqrt(24*total_days)*np.sqrt(24*365*n1)
    total_loss_fut_10 = loss_cyc_fut_10 + loss_cal_fut_10
    deg_rate_fut = np.median(np.diff(total_loss_fut_10,axis = 1),axis=1)/365 #Degradation rate is expressed per year, hence need to be divided by 365
    
    deg_cost_final = deg_rate_fut*c_new_batt
    
    
    
    
    
    return Batt_cap_degr_new, total_throughput_q_ch, total_throughput_q, loss_calendar, loss_cyclic_lt, loss_cyclic_ht, derivative_q_ch, derivative_q, deg_cost_final
    
    
    
    