from numba import njit
import numpy as np
import math

@njit
def degr_semi_empirical_PSO(Batt_cap_degr_old, batt_powers, batt_SOC , Batt_cap_full, 
                                  total_ch_old, total_q_old, total_days, last_der_q_ch, 
                                  last_der_q, last_loss_calendar, Ua_SOC_data, no_of_days_deg):
    
    temp = 23 + 273.15  # Kelvin
    Batt_Voltage = 51.2  # V
    R = 8.314
    t_ref = 298.15
    k_cal_ref = 3.694e-4
    Ea = 20592
    alpha = 0.384
    k0 = 0.142
    F = 96485
    ua_ref = 0.123
    
    # Calendar aging
    SOC_lookup = Ua_SOC_data[:, 0]
    n_particles, n_times = batt_SOC.shape
    Ua = np.empty((n_particles, n_times))
    f_SOC_cal = np.empty((n_particles, n_times))
    
    for i in range(n_particles):
        for j in range(n_times):
            target_SOC = batt_SOC[i, j] * 100
            
            # Start with the first value as "best"
            best_idx = 0
            best_diff = abs(SOC_lookup[0] - target_SOC)
            
            # Search for the closest match
            for k in range(1, len(SOC_lookup)):
                diff = abs(SOC_lookup[k] - target_SOC)
                if diff < best_diff:  # strictly smaller difference
                    best_diff = diff
                    best_idx = k
                # If diff == best_diff, keep the first index (same tie-breaking as np.argmin)
            Ua[i, j] = Ua_SOC_data[best_idx, 1]
            f_SOC_cal[i,j] = k0 + math.exp(alpha*F/R*(ua_ref - Ua[i,j])/t_ref)
    
    f_T_cal = math.exp(-Ea/R*(1/temp - 1/t_ref))
    k_cal = k_cal_ref * f_T_cal
    
    derivative_time = np.empty((n_particles, n_times))
    for i in range(n_particles):
        for j in range(n_times):
            derivative_time[i,j] = (k_cal * f_SOC_cal[i,j]*0.5) / math.sqrt((total_days-1)*24 + (j+1))
    
    loss_calendar = last_loss_calendar + np.sum(derivative_time, axis=1)
    
    # Cyclic aging
    k_cyc_lt_ref = 4.009e-4
    Ea_cyc_lt = 55546
    k_cyc_ht_ref = 1.456e-4
    Ea_cyc_ht = 32699
    
    cell_current_cyclic = batt_powers / Batt_Voltage * 1000 * (5/Batt_cap_full) * 3 / 70
    cell_current_ch_cyclic = np.zeros_like(cell_current_cyclic)
    
    for i in range(n_particles):
        for j in range(n_times-1):
            if cell_current_cyclic[i,j] < 0:
                cell_current_ch_cyclic[i,j] = -cell_current_cyclic[i,j]
            else:
                cell_current_ch_cyclic[i,j] = 0.0
    
    cell_throughput_total_ch = np.sum(np.abs(cell_current_ch_cyclic), axis=1)
    cell_throughput_total = np.sum(np.abs(cell_current_cyclic), axis=1)
    
    total_throughput_q_ch = cell_throughput_total_ch + total_ch_old
    total_throughput_q = cell_throughput_total + total_q_old
    
    k_cyc_lt = k_cyc_lt_ref * math.exp(Ea_cyc_lt/R * (1/temp - 1/t_ref))
    k_cyc_ht = k_cyc_ht_ref * math.exp(-Ea_cyc_ht/R * (1/temp - 1/t_ref))
    
    derivative_q = np.zeros(n_particles)
    derivative_q_ch = np.zeros(n_particles)
    
    for i in range(n_particles):
        if total_throughput_q[i] > 0:
            derivative_q[i] = k_cyc_ht/(2*math.sqrt(total_throughput_q[i]))
        if total_throughput_q_ch[i] > 0:
            derivative_q_ch[i] = k_cyc_lt/(2*math.sqrt(total_throughput_q_ch[i]))
    
    loss_cyclic_ht = np.empty(n_particles)
    loss_cyclic_lt = np.empty(n_particles)
    
    if total_days == 1:
        for i in range(n_particles):
            loss_cyclic_ht[i] = k_cyc_ht_ref * math.sqrt(total_throughput_q[i])
            loss_cyclic_lt[i] = k_cyc_lt_ref * math.sqrt(total_throughput_q_ch[i])
    else:
        for i in range(n_particles):
            loss_cyclic_ht[i] = last_der_q * cell_throughput_total[i]
            loss_cyclic_lt[i] = last_der_q_ch * cell_throughput_total_ch[i]
    
    degr_old = 1 - Batt_cap_degr_old/Batt_cap_full
    loss_total = (loss_calendar - last_loss_calendar + loss_cyclic_lt + loss_cyclic_ht)
    degr_new = degr_old + loss_total
    Batt_cap_degr_new = Batt_cap_full * (1 - degr_new)
    
    # Degradation cost
    Batt_cost_siz_energy = 500
    c_new_batt = Batt_cap_full * Batt_cost_siz_energy
    no_years = 10
    Usable_SOH = 0.2
    
    loss_cyc_fut_10 = np.zeros((n_particles, no_years))
    loss_cal_fut_10 = np.zeros((n_particles, no_years))
    
    for i in range(n_particles):
        for n1 in range(no_years):
            cyc_fut_ht = 0.0
            cyc_fut_lt = 0.0
            if total_throughput_q[i] > 0:
                cyc_fut_ht = loss_cyclic_ht[i]/math.sqrt(total_throughput_q[i])*math.sqrt(total_throughput_q[i]/total_days*365*n1)
            if total_throughput_q_ch[i] > 0:
                cyc_fut_lt = loss_cyclic_lt[i]/math.sqrt(total_throughput_q_ch[i])*math.sqrt(total_throughput_q_ch[i]/total_days*365*n1)
            loss_cyc_fut_10[i,n1] = cyc_fut_ht + cyc_fut_lt
            loss_cal_fut_10[i,n1] = loss_calendar[i]/math.sqrt(24*total_days)*math.sqrt(24*365*n1)
    
    n_particles, no_years = loss_cyc_fut_10.shape
    deg_rate_fut = np.empty(n_particles)

    for i in range(n_particles):
        # Collect differences into a temporary array
        diffs = np.empty(no_years - 1)
        for j in range(1, no_years):
            diffs[j - 1] = (loss_cyc_fut_10[i, j] + loss_cal_fut_10[i, j]) - \
                           (loss_cyc_fut_10[i, j - 1] + loss_cal_fut_10[i, j - 1])

        # Compute median manually
        sorted_diffs = np.sort(diffs)
        m = no_years - 1
        if m % 2 == 1:
            median_val = sorted_diffs[m // 2]
        else:
            median_val = 0.5 * (sorted_diffs[m // 2 - 1] + sorted_diffs[m // 2])

        deg_rate_fut[i] = median_val / 3650.0
    
    deg_cost_final = deg_rate_fut * c_new_batt / Usable_SOH
    
    return Batt_cap_degr_new, total_throughput_q_ch, total_throughput_q, loss_calendar, loss_cyclic_lt, loss_cyclic_ht, derivative_q_ch, derivative_q, deg_cost_final
