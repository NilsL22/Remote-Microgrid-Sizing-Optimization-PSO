# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 14:04:42 2025

@author: nilsl
"""
import numpy as np
from numba import njit, prange
from math import pow, exp
import rainflow


#@njit
def reversals_numba(series):
    n = len(series)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)  # no reversals
    
    # pre-allocate max size (worst case: all points are reversals)
    revs = np.empty((n, 2), dtype=np.float64)
    count = 0

    # first point is always a reversal
    revs[count, 0] = 0
    revs[count, 1] = series[0]
    count += 1

    d_last = series[1] - series[0]
    for i in range(1, n - 1):
        d_next = series[i+1] - series[i]
        if d_last * d_next < 0:
            revs[count, 0] = i
            revs[count, 1] = series[i]
            count += 1
        d_last = d_next

    # last point always a reversal
    revs[count, 0] = n - 1
    revs[count, 1] = series[n - 1]
    count += 1

    return revs[:count]


#@njit
def extract_cycles_numba(series):
    rev_points = reversals_numba(series)
    n = len(rev_points)
    
    # worst case all points in half cycles + some full cycles
    max_cycles = n * 2  
    cycles = np.empty((max_cycles, 5), dtype=np.float64)  # range, mean, count, start_idx, end_idx
    count = 0
    
    # Use fixed-size stack for points indices
    stack_i = np.empty(n, dtype=np.int64)
    stack_v = np.empty(n, dtype=np.float64)
    s = 0  # stack pointer
    
    for idx in range(n):
        stack_i[s] = int(rev_points[idx, 0])
        stack_v[s] = rev_points[idx, 1]
        s += 1
        
        while s >= 3:
            x1, x2, x3 = stack_v[s-3], stack_v[s-2], stack_v[s-1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            
            if X < Y:
                break
            elif s == 3:
                # half cycle on first two points
                rng = abs(stack_v[0] - stack_v[1])
                mean = 0.5 * (stack_v[0] + stack_v[1])
                cycles[count, 0] = rng
                cycles[count, 1] = mean
                cycles[count, 2] = 0.5
                cycles[count, 3] = stack_i[0]
                cycles[count, 4] = stack_i[1]
                count += 1
                
                # discard first point
                for k in range(s - 1):
                    stack_i[k] = stack_i[k+1]
                    stack_v[k] = stack_v[k+1]
                s -= 1
            else:
                # full cycle on points s-3 and s-2
                rng = abs(stack_v[s-3] - stack_v[s-2])
                mean = 0.5 * (stack_v[s-3] + stack_v[s-2])
                cycles[count, 0] = rng
                cycles[count, 1] = mean
                cycles[count, 2] = 1.0
                cycles[count, 3] = stack_i[s-3]
                cycles[count, 4] = stack_i[s-2]
                count += 1
                
                # remove points s-3 and s-2, keep s-1
                # move s-1 to s-3 pos
                stack_i[s-3] = stack_i[s-1]
                stack_v[s-3] = stack_v[s-1]
                s -= 2

    # remaining half cycles
    for i in range(s - 1):
        rng = abs(stack_v[i] - stack_v[i+1])
        mean = 0.5 * (stack_v[i] + stack_v[i+1])
        cycles[count, 0] = rng
        cycles[count, 1] = mean
        cycles[count, 2] = 0.5
        cycles[count, 3] = stack_i[i]
        cycles[count, 4] = stack_i[i+1]
        count += 1
        
    return cycles[:count,3], cycles[:count,4], count






#@njit(parallel = True)
def compute_v_index_fast(P_actual_rel, PV_curves_rel):
    n_samples, n_curves = P_actual_rel.shape
    n_voltages = PV_curves_rel.shape[1]
    V_index = np.empty((n_samples, n_curves), dtype=np.int32)

    for i in prange(n_samples):
        for j in prange(n_curves):
            curve = PV_curves_rel[j]
            target = P_actual_rel[i, j]

            # Binary search
            low = 0
            high = n_voltages - 1
            while low < high:
                mid = (low + high) // 2
                if curve[mid] < target:
                    low = mid + 1
                else:
                    high = mid
            # Choose closer index
            if low > 0 and abs(curve[low-1] - target) < abs(curve[low] - target):
                V_index[i, j] = low - 1
            else:
                V_index[i, j] = low
    return V_index



#@njit(parallel = True)
def extract_PV_curves_from_LUT(T_profile, G_profile, PV_LUT_loc):
    T_range = np.linspace(15.0, 34.5, 40)
    G_range = np.linspace(0.2, 0.99, 80)
    
    # Temperature index
    T_idx = np.searchsorted(T_range, T_profile)
    T_idx = np.clip(T_idx, 0, len(T_range) - 1)
    T_diffs = np.abs(T_range[T_idx] - T_profile)
    
    T_idx_adj = np.empty_like(T_idx)
    for i in range(T_idx.shape[0]):
        if (T_idx[i] > 0 and 
            T_diffs[i] > abs(T_range[T_idx[i] - 1] - T_profile[i])):
            T_idx_adj[i] = T_idx[i] - 1
        else:
            T_idx_adj[i] = T_idx[i]
    
    # Irradiance index
    G_idx = np.searchsorted(G_range, G_profile)
    G_idx = np.clip(G_idx, 0, len(G_range) - 1)
    G_diffs = np.abs(G_range[G_idx] - G_profile)
    
    G_idx_adj = np.empty_like(G_idx)
    for i in range(G_idx.shape[0]):
        if (G_idx[i] > 0 and 
            G_diffs[i] > abs(G_range[G_idx[i] - 1] - G_profile[i])):
            G_idx_adj[i] = G_idx[i] - 1
        else:
            G_idx_adj[i] = G_idx[i]
    
    # Extract PV curves
    n_points = PV_LUT_loc.shape[2]
    PV_curves1 = np.empty((len(T_idx_adj), n_points), dtype=PV_LUT_loc.dtype)
    for i in range(len(T_idx_adj)):
        PV_curves1[i, :] = PV_LUT_loc[T_idx_adj[i], G_idx_adj[i], :]
    
    
    # Compute max per row manually (Numba safe)
    n_rows, n_cols = PV_curves1.shape
    max_vals = np.empty(n_rows, dtype=PV_curves1.dtype)
    for i in range(n_rows):
        row_max = PV_curves1[i, 0]
        for j in range(1, n_cols):
            if PV_curves1[i, j] > row_max:
                row_max = PV_curves1[i, j]
        max_vals[i] = row_max
    
    
    
    # Normalize
    PV_curves_rel = np.empty_like(PV_curves1, dtype=np.float32)
    for i in range(PV_curves1.shape[0]):
        if max_vals[i] != 0:
            PV_curves_rel[i, :] = PV_curves1[i, :] / max_vals[i]
        else:
            PV_curves_rel[i, :] = 0.0
    
    return PV_curves_rel




    
#@njit(parallel=True)
def degr_switch(P_PV_total, P_PV_dispatched, Vin_sw, PV_size, T_amb):
    A = 9.34*10**14
    b1, b2, b3, b4, b5, b6 = -4.416, 1285, -0.463, -0.716, -0.761, -0.5
    I_b = 10
    V_b = 600 / 100
    d = 450*10**(-6)
    R_CE = 0.0856
    VT = 1.198
    V_out = 80 # proportionally to 400V to be representative to one panel
    L = 0.00145
    fsw = 20*10**3
    dt = 3600
    
    
    if PV_size > 0:
        scale_factor = 400 / (PV_size*1000)
    else:
        scale_factor = 0
    dynamics = np.array([0.0195, 0.011, 0.0005])
    R_th_sum = 11.03
    
    n_rows, n_cols = Vin_sw.shape
    inverter_cost = 500 * PV_size #The one in Samionta costs somewhere between 2000 and 2500
    D = 1 - Vin_sw/ V_out
    P_PV_total = P_PV_total * scale_factor*1000
    P_PV_dispatched = P_PV_dispatched * scale_factor*1000

    Iav = np.zeros((n_rows, n_cols))
    Isw = np.zeros((n_rows, n_cols))
    
    for i in prange(n_rows):
        for j in range(n_cols):
            v = Vin_sw[i, j]
            if v > 0:
                Iav[i, j] = P_PV_total[j] * D[i, j] / v
                Isw[i, j] = (D[i, j] * (P_PV_dispatched[i, j] / v) ** 2 +
                             (1 / 3) * (v * D[i, j] / (2 * fsw * L))) ** 0.5
    
    E_cond = VT * Iav + R_CE * Isw ** 2
    E_sw = fsw * (dynamics[0] + dynamics[1] * Iav + dynamics[2] * Isw ** 2) / 1000
    T_switch = np.clip((E_cond + E_sw) * R_th_sum + T_amb,0,175)
    #T_switch = np.array([np.load("T_sw_final_test.npy")[0,:]])
    T_switch_new = np.empty((T_switch.shape[0],T_switch.shape[1]+1))
    T_switch_new[:,1:] = T_switch
    T_switch_new[:,0] = 25.0
    
    degr_sw_day = np.zeros(n_rows)
    degr_cost_sw = np.zeros(n_rows)

    rf_starts = np.empty((n_rows, 24), dtype=np.int64)
    rf_ends = np.empty((n_rows, 24), dtype=np.int64)
    counts = np.empty(n_rows, dtype=np.int64)
    
    damage_multiplier = 6023/15 #to relate the damage to realistic numbers -> based on Joel's work
    for i in prange(n_rows):
        starts_numba, ends_numba, count_numba = extract_cycles_numba(T_switch_new[i, :])
        rf_starts[i, :count_numba] = starts_numba
        rf_ends[i, :count_numba] = ends_numba
        counts[i] = count_numba

        for k in range(count_numba):
            s = rf_starts[i, k]
            e = rf_ends[i, k]
            #T_min = T_switch[i, s]+273.15 if T_switch[i, s] < T_switch[i, e] else T_switch[i, e]+273.15
            T_min = min(T_switch_new[i,s:e])+273.15
            T_max = max(T_switch_new[i,s:e])+273.15
            delta_T = T_max-T_min
            #delta_T = abs(T_switch[i, e] - T_switch[i, s])
            t_heat_on = dt * (e - s)
            if delta_T > 0 and t_heat_on > 0:
                degr_sw_day[i] += damage_multiplier / (A * pow(delta_T, b1) * exp(b2 / T_min) *
                                         pow(t_heat_on, b3) * pow(I_b, b4) *
                                         pow(V_b, b5) * pow(d, b6))
            else:
                degr_sw_day[i] += 0
        degr_cost_sw[i] = degr_sw_day[i] * inverter_cost
             

    return degr_sw_day, degr_cost_sw, T_switch






