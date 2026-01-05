import os
import re
import numpy as np
import pymbar
import pickle
import matplotlib.pyplot as plt
import glob
# 
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp

######
#Ising
######

def p_ising_theoretical(x):
    a = 0.158
    c = 0.776
    M0 = 1.1341655
    term = (x**2 / M0**2)
    exponent = -1.0 * ((term - 1.0)**2) * (a * term + c)
    prob = np.exp(exponent)
    return prob / np.sum(prob)
    
##############
#ising fiting#
##############

class IsingFitter:
    def __init__(self, mbar, E_all, V_all, N_k, PV_FACTOR, KB=0.008314462):
        self.mbar = mbar
        self.E_all = E_all
        self.V_all = V_all
        self.rho_all = 8.9745 / V_all
        self.N_k = N_k
        self.PV_FACTOR = PV_FACTOR
        self.KB = KB
        self.f_k = mbar.f_k
        self.u_kn = mbar.u_kn
        self.bins = 200
        self.range_M = (-4, 4)
        self.bin_edges = np.linspace(self.range_M[0], self.range_M[1], self.bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.p_theory = p_ising_theoretical(self.bin_centers)
        self.p_theory /= (np.sum(self.p_theory) * (self.bin_centers[1]-self.bin_centers[0]))

    def get_weights(self, T, P):
        beta = 1.0 / (self.KB * T)
        u_target = beta * (self.E_all + P * self.V_all * self.PV_FACTOR)
        log_terms = self.f_k[:, np.newaxis] - self.u_kn
        log_denom = logsumexp(log_terms, b=self.N_k[:, np.newaxis], axis=0)
        log_w = -u_target - log_denom
        max_log_w = np.max(log_w)
        weights = np.exp(log_w - max_log_w)
        weights /= np.sum(weights)
        return weights

    def loss_function(self, params):
        T, P, s = params
        # 
        if not (170 < T < 200) or not (1000 < P < 3000):
            return 1e9
        
        try:
            weights = self.get_weights(T, P)
            
            # 
            # 
            neff = 1.0 / np.sum(weights**2)
            if neff < 50: 
                return 1e9 # 
                
        except:
            return 1e9
            
        t = self.rho_all + s * (self.E_all/300)
        mean_t = np.average(t, weights=weights)
        var_t = np.average((t - mean_t)**2, weights=weights)
        
        if var_t < 1e-12: return 1e9 # 
        
        std_t = np.sqrt(var_t)
        M = (t - mean_t) / std_t
        hist, _ = np.histogram(M, bins=self.bin_edges, weights=weights, density=True)
        error = np.sum((hist - self.p_theory)**2)
        return error

############

DISCARD_START = 100
GMX_COLUMN_IDX = 1
KB = 0.008314462     # kJ/(mol*K)
PV_FACTOR = 0.0602214086 
pattern = re.compile(r"P([\d\.]+)_T([\d\.]+)\.(Potential|Volume)\.xvg")
sim_files = {}
for filename in os.listdir('.'):
    match = pattern.match(filename)
    if match:
        p_str, t_str, file_type = match.groups()
        P_val = float(p_str)
        T_val = float(t_str)
        key = (P_val, T_val)
        if key not in sim_files: sim_files[key] = {}
        sim_files[key][file_type] = filename

valid_states = []
for (P, T), files in sim_files.items():
    if 'Potential' in files and 'Volume' in files:
        valid_states.append({'P': P, 'T': T, 'file_U': files['Potential'], 'file_V': files['Volume']})

valid_states.sort(key=lambda x: (x['P'], x['T']))
print(f"Find {len(valid_states)} ")

E_data = []
V_data = []
N_k = []
T_list = []
P_list = []

Q1_data = []
Q2_data = []
Q3_data = []

for state in valid_states:
    try:
        raw_u = np.loadtxt(state['file_U'], comments=['#', '@'])
        u_traj = raw_u[DISCARD_START:, GMX_COLUMN_IDX]
        raw_v = np.loadtxt(state['file_V'], comments=['#', '@'])
        v_traj = raw_v[DISCARD_START:, GMX_COLUMN_IDX]
        min_len = min(len(u_traj), len(v_traj))
        if len(u_traj) != len(v_traj):
            u_traj = u_traj[:min_len]
            v_traj = v_traj[:min_len]
        E_data.append(u_traj)
        V_data.append(v_traj)
        N_k.append(min_len)
        T_list.append(state['T'])
        P_list.append(state['P'])
        print(f"load: P={state['P']:<6} T={state['T']:<6} | N={min_len}")
    except Exception as e:
        print(f"error {state}: {e}")

N_k = np.array(N_k)
T_list = np.array(T_list)
P_list = np.array(P_list)
K_states = len(N_k)
E_all = np.concatenate(E_data)
V_all = np.concatenate(V_data)
N_total = np.sum(N_k)
print(f"--- Total: {N_total} ---")

u_kn = np.zeros((K_states, N_total))
for k in range(K_states):
    beta_k = 1.0 / (KB * T_list[k])
    p_k = P_list[k]
    u_kn[k, :] = beta_k * (E_all + p_k * V_all * PV_FACTOR)

mbar_filename = 'mbar_model.pkl'
mbar = None
if os.path.exists(mbar_filename):
    try:
        with open(mbar_filename, 'rb') as f: mbar = pickle.load(f)
        print("MBAR succeed")
    except: mbar = None

if mbar is None:
    print("Cal MBAR...")
    mbar = pymbar.MBAR(u_kn, N_k)
    with open(mbar_filename, 'wb') as f: pickle.dump(mbar, f)

print("MBAR finish ...")

#################################################################################################

print("\n--- Find LLCP (Global Search with Differential Evolution) ---")
fitter = IsingFitter(mbar, E_all, V_all, N_k, PV_FACTOR)

# 
bnds = [(178, 200), (1250, 2000), (-0.07, 0.07)]

# 
print("Waitting）...")
res = differential_evolution(
    fitter.loss_function, 
    bounds=bnds, 
    strategy='best1bin',
    maxiter=1000,
    popsize=50,
    tol=0.001,
    workers=-1,  # 
    disp=True
)

Tc_fit, Pc_fit, s_fit = res.x
print(f"\nfiting results:")
print(f"Critical Temperature (Tc) = {Tc_fit:.4f} K")
print(f"Critical Pressure (Pc)    = {Pc_fit:.4f} bar")
print(f"Mixing Parameter (s)      = {s_fit:.6f}")
print(f"Final Loss                = {res.fun:.6e}")

# 
weights_opt = fitter.get_weights(Tc_fit, Pc_fit)
t_opt = fitter.rho_all + s_fit * (fitter.E_all / 300.0)
mean_t = np.average(t_opt, weights=weights_opt)
std_t = np.sqrt(np.average((t_opt - mean_t)**2, weights=weights_opt))
M_opt = (t_opt - mean_t) / std_t

hist_opt, _ = np.histogram(M_opt, bins=fitter.bin_edges, weights=weights_opt, density=True)

plt.figure(figsize=(8, 6))
plt.plot(fitter.bin_centers, fitter.p_theory, 'k-', lw=2, label='3D-Ising ideal')
plt.plot(fitter.bin_centers, hist_opt, 'r--o', label=f'Sim Data @ {Tc_fit:.1f}K, {Pc_fit:.0f}bar')
plt.xlabel('Order Parameter M')
plt.ylabel('Probability Density P(M)')
plt.title(f'Ising Fit Result\nTc={Tc_fit:.2f} K, Pc={Pc_fit:.0f} bar, s={s_fit:.5f}, loss={res.fun:.6e}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'Ising_Fit_Result_{res.fun:.6e}.png', dpi=300)
plt.show()

data_fit = np.column_stack((fitter.bin_centers, hist_opt, fitter.p_theory))
np.savetxt(f"Ising_Fit_Data_Tc{Tc_fit:.2f}_Pc{Pc_fit:.0f}_{res.fun:.6e}.txt", 
           data_fit, 
           header="M_center  P_simulated  P_Ising_theoretical")

