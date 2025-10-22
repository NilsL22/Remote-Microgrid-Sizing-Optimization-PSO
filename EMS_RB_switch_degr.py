# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 07:05:49 2025

@author: nilsl
"""

num_threads = 1
# =============================================================================
# import os
# os.environ["OMP_NUM_THREADS"] = str(num_threads)       # NumPy / MKL / OpenBLAS threads
# os.environ["NUMBA_NUM_THREADS"] = str(num_threads)      # Numba threads
# 
# import numba
# numba.set_num_threads(num_threads)                  # Explicit Numba thread setting
# =============================================================================

from Switch_degr_func_RB import extract_PV_curves_from_LUT, degr_switch, compute_v_index_fast

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Schimpe_degradation_new_v5_daily import degr_semi_empirical_PSO
import datetime
import rainflow

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
        deg_rate_fut = np.median(np.diff(total_loss_over_years))
        linearized_semi_emp  = np.arange(0,3650*deg_rate_fut,deg_rate_fut)
        
        if make_plots == "yes":
            tick_labels = np.arange(1,11,2)
            plt.plot(np.arange(365*no_years), loss_cyc_over_years)
            plt.plot(np.arange(365*no_years), loss_cal_over_years)
            plt.xticks(np.arange(0,3650,365*2),tick_labels)
            plt.legend(["Cyclic Degradation", "Calendar Degradation"], fontsize = 12)
            plt.ylim((0,0.2))
            #plt.title("Degradation by its type")
            plt.xlabel("Time [Years]", fontsize = 13)
            plt.ylabel("Degradation [%]", fontsize = 13)
            plt.savefig("Cal_cyc_degr.eps", format = 'eps')
            plt.show()
            plt.plot(np.arange(365*no_years),total_loss_over_years)
            #plt.plot(np.arange(365*no_years),linearized_semi_emp)
            plt.xticks(np.arange(0,3650,365*2),tick_labels)
            #plt.title("Total Degradation")
            
            simple_deg_day_3000 = 0.006879/100
            simple_deg_over_time_3000 = np.arange(0,simple_deg_day_3000*365*no_years,simple_deg_day_3000)
            plt.plot(np.arange(365*no_years),simple_deg_over_time_3000)
            
            simple_deg_day_4500 = 0.00516/100
            simple_deg_over_time_4500 = np.arange(0,simple_deg_day_4500*365*no_years,simple_deg_day_4500)
            plt.plot(np.arange(365*no_years),simple_deg_over_time_4500)
            
            simple_deg_day_6000 = 0.00344/100
            simple_deg_over_time_6000 = np.arange(0,simple_deg_day_6000*365*no_years,simple_deg_day_6000)
            plt.plot(np.arange(365*no_years),simple_deg_over_time_6000)
            
            
            
            plt.legend(["SE Model", "ET Model: 3000 Cycles", "ET Model: 4500 Cycles", "ET Model: 6000 Cycles"], fontsize = 12)
            plt.xlabel("Time [Years]", fontsize = 13)
            plt.ylabel("Degradation [%]", fontsize = 13)
            plt.ylim((0,0.3))
            plt.savefig("ET-SE_comp.eps", format = 'eps')
            plt.show()
            
        return deg_cost_fut, loss_cyc_fut, loss_cal_fut, total_loss_fut, total_loss_over_years




def calc_degr(Batt_powers,Batt_cap_siz_max_SOC, Batt_cap_degr_loc): #energy-throughput model
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
    c_new_batt = Batt_cap_siz_max_SOC*Batt_cost_siz_energy
    #degr_rate = Usable_SOH/RUL
    #degr_cost = degr_rate*(c_new_batt - c_salv_batt) #accumulative cost -> cannot use for each cycle separately
    degr_cost = SOH_delta*c_new_batt
    return Batt_cap_new, degr_cost

def dispatch_RB(sizing_limits, PV_pow_max_rb, Load_rb_loc, degr_type, Ua_SOC_data_rb, T_profile_RB, G_profile_RB, PV_LUT_RB, allow_sw_degr, Gen_cost_curve_RB):
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
    P_curtailed = np.zeros(Load_rb_loc.shape) 
    Power_balance = np.zeros(Load_rb_loc.shape)
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
            E_load_left = np.sum(Load_rb_loc[t1,t2:min(t2+forecast_horizon, Load_rb_loc.shape[1]-t2)])
            P_batt_max_ch = min(Batt_max_pow_rb, (Batt_cap_rb[t1]*SOC_limit_high/100-E_batt_left)/time_step)
            P_batt_max_dis = min(Batt_max_pow_rb, ((Prev_SOC-SOC_limit_low)/100)*Batt_cap_rb[t1]/time_step)
            if PV_pow_max_rb[t1,t2] > 0:
                if PV_pow_max_rb[t1,t2] < Load_rb_loc[t1,t2]:    
                    if ((E_batt_left-SOC_limit_low/100*Batt_cap_rb[t1]) > E_load_left) and (P_batt_max_dis > Load_rb_loc[t1,t2]):
                        P_gen[t1,t2] = 0
                        P_batt[t1,t2] = Load_rb_loc[t1,t2] - PV_pow_max_rb[t1,t2]
                        P_dump[t1,t2] = 0
                    else:
                        P_gen[t1,t2] = min(Gen_pow_max, (Load[t1,t2] - PV_pow_max_rb[t1,t2] + P_batt_max_ch))
                        P_batt[t1,t2] = -(P_gen[t1,t2] - (Load[t1,t2] - PV_pow_max_rb[t1,t2]))
                        P_dump[t1,t2] = 0
                else:
                    P_gen[t1,t2] = 0
                    P_batt[t1,t2] = -min((PV_pow_max_rb[t1,t2] - Load_rb_loc[t1,t2]), P_batt_max_ch)
                    P_dump[t1,t2] = PV_pow_max_rb[t1,t2] + P_batt[t1,t2] - Load_rb_loc[t1,t2]
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
        P_PV_dispatched = PV_pow_max_rb - P_dump
        if degr_type == "semi_emp":
            Batt_cap_rb[t1+1], total_ch_new, total_q_new, loss_calendar_new, loss_cyclic_lt_new, loss_cyclic_ht_new, derivative_q_ch_new, derivative_q_new, degr_cost_tot_batt_RB[t1] = degr_semi_empirical_PSO(Batt_cap_rb[t1], np.array([P_batt[t1]]), np.array([SOC[t1]/100]) , Batt_cap_max_dispatch, total_ch_over_time_rb[t1], total_q_over_time_rb[t1], t1+1, der_qch_over_time_rb[t1], der_q_over_time_rb[t1], loss_cal_over_time_rb[t1], Ua_SOC_data_rb, no_of_days) 
            
            total_ch_over_time_rb[t1+1] = total_ch_new
            total_q_over_time_rb[t1+1] = total_q_new
            der_qch_over_time_rb[t1+1] = derivative_q_ch_new
            der_q_over_time_rb[t1+1] = derivative_q_new
            loss_cal_over_time_rb[t1+1] = loss_calendar_new
            loss_cyc_lt_over_time_rb[t1+1] = loss_cyclic_lt_new
            loss_cyc_ht_over_time_rb[t1+1] = loss_cyclic_ht_new
        else:
            Batt_cap_rb[t1+1], degr_cost_tot_batt_RB[t1] = calc_degr(np.array([P_batt[t1]]), Batt_cap_max_dispatch, Batt_cap_rb[t1]) #P_batt is put inside another matrix to match with the form of the calc_degr function
            
        
        P_actual_rel = np.zeros((1,24))
        for m in range(24):
            if P_PV_dispatched[0,m] > 0:
                P_actual_rel[0,m] = P_PV_dispatched[t1,m] / PV_pow_max_rb[t1,m]
        Vin_sw = compute_v_index_fast(P_actual_rel, PV_curves_rel_RB)
        rel_switch_degr_RB[t1], degr_cost_tot_sw_RB[t1], T_switch_RB[t1,:] = degr_switch(PV_pow_max_rb[t1,:], P_actual_rel, Vin_sw, sizing_limits[1], T_profile_day)
        
        
        if allow_sw_degr == "no":
            degr_cost_tot_sw_RB[t1] = 0
            
        
    #Power_balance = PV_pow_max_rb + P_batt + P_gen - Load_rb_loc - P_dump #just for debugging
    fuel_rel_output = P_gen/np.max(Load_rb_loc)
    fuel_cost = np.interp(fuel_rel_output,Gen_cost_curve_RB[:,0],Gen_cost_curve_RB[:,1]) #the cost of the fuel depends on the operating point inside the generator efficiency curve
    opex = np.sum(fuel_cost*P_gen) + np.sum(degr_cost_tot_batt_RB) + np.sum(degr_cost_tot_sw_RB)
    #opex = fuel_cost*P_gen + degr_cost_tot_batt_RB + degr_cost_tot_sw_RB
    power_flows = np.stack((P_PV_dispatched,P_batt,P_gen),axis = -1)
    fuel_cost_RB = np.sum(fuel_cost*P_gen)
    
    return opex, power_flows, SOC, Batt_cap_rb, P_dump, fuel_cost_RB, total_ch_over_time_rb[-1], total_q_over_time_rb[-1], loss_cal_over_time_rb[-1], np.sum(loss_cyc_lt_over_time_rb), np.sum(loss_cyc_ht_over_time_rb), rel_switch_degr_RB, T_switch_RB
    #return opex, power_flows, SOC, Batt_cap_rb, P_dump, fuel_cost_RB, total_ch_over_time_rb, total_q_over_time_rb, loss_cal_over_time_rb, loss_cyc_lt_over_time_rb, loss_cyc_ht_over_time_rb, rel_switch_degr_RB, T_switch_RB

#Solar Power - aquired from renewables.ninja in Benin 
no_of_days = 40
#pv_power_ready = np.load("pv_power_ready_new_2_morning.npy")
#pv_power = pv_power_ready[:no_of_days]
pv_power = np.load("PV_profile_40.npy")[:no_of_days,:]
PV_LUT = np.load("PV_LUT_3.npy")
G_profile = np.load("Irradiance_profile_40.npy")[:no_of_days,:]
T_profile_rounded = np.round(np.load("Temperature_profile_40.npy")*2)/2
T_profile = T_profile_rounded[:no_of_days,:]



#Load - aquired by using RAMP
lifetime = 20
Load_ready = np.load("repr_days_load_morning.npy") 
#Load_ready = np.load("Load_smooth.npy")
#Load_ready = np.tile(Load_ready,(2,1))
Load_growth = 1.05 #per year
Load_multiplier_20 = Load_growth**20
#Load = Load_ready[:no_of_days,:]/1000*Load_multiplier_20 #Year 20 -> change back to basic Load and implement the growth in the sizing func
Load = np.zeros((no_of_days,24))
Load[:int(no_of_days/2),:] = Load_ready[:int(no_of_days/2),:]
Load[int(no_of_days/2):no_of_days,: ] = Load_ready[20:20+int(no_of_days/2),:] 
Load = Load/1000*Load_multiplier_20
Gen_pow_max = np.max(Load) #kW
time_step = 1 #in hours
no_time_instances = 24
#Fuel consumption curve
Diesel_price_lit = 0.7*1.2 #Diesel_price + transportation cost
Fuel_eff_curve = np.load("Fuel_efficiency_curve_L_kWh.npy")
Gen_cost_curve = np.zeros((1000,2))
Gen_cost_curve[:,0] = np.linspace(0,1,1000)
Gen_cost_curve[:,1] = Fuel_eff_curve*Diesel_price_lit
Gen_cost_curve[0] = 0 #The pso version has the first values at very high cost, which is not the case here
PV_cost_EMS = 0 
Batt_cost_siz_energy = 500 #per kWh from solartopstore; 
Batt_cost_siz_power = 0 #per kW
Batt_cost_EMS = 0 #in principle 0 unless some kind of maintanence is included or the additional energy of the AC/ degr cost is calculated separately
#Battery 
#Batt_pow_max = 3.5 #kW
Batt_SOC = 40 #%
eff_ch = 0.98
eff_dis = 0.98
Batt_cap_max = 40
pow_cap_ratio = 0.35


Ua_SOC_data = pd.read_csv("Anode_voltage_vs_SOC.csv").to_numpy()
degr_type_final = "semi_emp"
#degr_type_final = "simple"

allow_sw_degr_final = "yes"
#allow_sw_degr_final = "no"
Final_sys_size_20 = np.array([29.1,6]) #Ebatt, P_pv
pv_power_rb_final = pv_power*Final_sys_size_20[1]
start_time = time.perf_counter()
opex_final, pos_EMS_over_time_final, SOC_over_time_final, Batt_cap_degr_over_time_final, P_curt_siz_final, fuel_cost_siz_final, total_ch_final, total_q_final, loss_cal_final, loss_cyc_lt_final, loss_cyc_ht_final, rel_switch_degr_final, T_sw_over_time_final = dispatch_RB(Final_sys_size_20,pv_power_rb_final,Load, degr_type_final, Ua_SOC_data, T_profile, G_profile, PV_LUT, allow_sw_degr_final, Gen_cost_curve)
end_time = time.perf_counter()
print("Time taken: ",end_time-start_time)

# =============================================================================
# loss_cyc_fut_10 = np.zeros(10)
# loss_cal_fut_10 = np.zeros(10)
# total_loss_fut_10_2 = np.zeros(no_of_days)
# deg_rate_fut = np.zeros(no_of_days)
# deg_cost_fut = np.zeros(no_of_days)
# 
# loss_cyc_ht_final_2 = np.cumsum(loss_cyc_ht_final)
# loss_cyc_lt_final_2 = np.cumsum(loss_cyc_lt_final)
# 
# for n0 in range(1,no_of_days):
#     for n1 in range(10):
#         cyc_fut_ht = loss_cyc_ht_final_2[n0]/np.sqrt(total_q_final[n0])*np.sqrt(total_q_final[n0]/n0*365*n1)
#         cyc_fut_lt = loss_cyc_lt_final_2[n0]/np.sqrt(total_ch_final[n0])*np.sqrt(total_ch_final[n0]/n0*365*n1)
#         loss_cyc_fut_10[n1] = cyc_fut_ht + cyc_fut_lt
#         loss_cal_fut_10[n1] = loss_cal_final[n0]/np.sqrt(24*n0)*np.sqrt(24*365*n1)
#     total_loss_fut_10 = loss_cyc_fut_10 + loss_cal_fut_10
#     #total_loss_fut_10_2[n0] = total_loss_fut_10[-1]
#     deg_rate_fut[n0] = np.median(np.diff(total_loss_fut_10))/365
#     deg_cost_fut[n0] = deg_rate_fut[n0]*500*20/0.2
#     plt.plot(total_loss_fut_10)
# plt.show()
# 
# plt.plot(deg_rate_fut[1:])
# plt.plot(np.arange(len(deg_rate_fut-1)),np.ones(len(deg_rate_fut-1))*np.mean(deg_rate_fut[1:]))
# plt.show()
# 
# =============================================================================


print("Opex per day:", opex_final/no_of_days)
if degr_type_final == "semi_emp":
    no_of_years = 10
    deg_cost_2, loss_cyc_2, loss_cal_2, total_loss_2, total_loss_over_years_final = semi_emp_degr_future(Batt_cap_degr_over_time_final[-1], no_of_days, total_ch_final, total_q_final, no_of_years, loss_cyc_lt_final, loss_cyc_ht_final, loss_cal_final, Final_sys_size_20[0], "yes")
    print("Battery Throughput per day: ", np.sum(abs(pos_EMS_over_time_final[:,:,1]))/no_of_days)
    print("Generator Energy per day: ", np.sum(pos_EMS_over_time_final[:,:,2])/no_of_days)
    print("Total Degradation: ", total_loss_2)
    print("Total Switch Degradation: ", np.sum(rel_switch_degr_final)/no_of_days*3650)
else:
    daily_loss = (1-Batt_cap_degr_over_time_final[-1]/Batt_cap_degr_over_time_final[0])/no_of_days
    print("Daily Loss: ", daily_loss)
    total_loss_simple = (1 - Batt_cap_degr_over_time_final[-1]/Batt_cap_degr_over_time_final[0])/no_of_days*3650
    print("Battery Throughput per day: ", np.sum(abs(pos_EMS_over_time_final[:,:,1]))/no_of_days)
    print("Generator Energy per day: ", np.sum(pos_EMS_over_time_final[:,:,2])/no_of_days)
    print("Total Battery Degradation: ", total_loss_simple)
    print("Total Switch Degradation: ", np.sum(rel_switch_degr_final)/no_of_days*3650)
    print("Curtailed power per day:, ", np.sum(P_curt_siz_final)/no_of_days)
# =============================================================================
## Assuming pos_EMS_over_time_final is already defined and has shape (samples, time_points, 3)
#time_axis = [datetime.datetime(2024, 1, 1, 8) + datetime.timedelta(hours=i) for i in range(24)]  # 08:00 to 08:00 next day
#tick_times = [time_axis[0] + datetime.timedelta(hours=i) for i in range(0, 24, 2)]
# plt.plot(time_axis, np.mean(pos_EMS_over_time_final[:,:,0], axis=0), label='PV', color = 'orange')
# plt.plot(time_axis, np.mean(pos_EMS_over_time_final[:,:,1], axis=0), label='Battery', color = 'royalblue')
# plt.plot(time_axis, np.mean(pos_EMS_over_time_final[:,:,2], axis=0), label='Generator', color = 'brown')
# 
# plt.xticks(tick_times, [dt.strftime('%H:%M') for dt in tick_times], rotation=45)
# plt.xlabel("Time of Day")
# plt.ylabel("Mean EMS Position")
# plt.title("EMS Position Over Time")
# plt.legend()
# plt.tight_layout()
# plt.show()
# 
# =============================================================================




