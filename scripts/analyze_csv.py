import pandas as pd
import numpy as np

df = pd.read_csv('d:/MRM_Inverse_Design_V1.1/data/optimization_results.csv')
total = len(df)
print(f"Total samples: {total}")

# Current constraint thresholds
ER_min = 4.0
Q_min = 4000.0
Q_max = 10000.0
fEO_min = 25.0  # GHz
FSR_min = 6.4   # nm

# Evaluate conditions
er_fail = df['ER (dB)'] < ER_min
q_lower_fail = df['Q Factor'] < Q_min
q_upper_fail = df['Q Factor'] > Q_max
q_fail = q_lower_fail | q_upper_fail
feo_fail = df['f_EO (GHz)'] < fEO_min
fsr_fail = df['FSR (nm)'] < FSR_min

print("\n--- Failure Attribution ---")
print(f"ER Failure (< {ER_min} dB): {er_fail.sum()} ({er_fail.sum()/total*100:.1f}%)")
print(f"Q Factor Failure (not in {Q_min}-{Q_max}): {q_fail.sum()} ({q_fail.sum()/total*100:.1f}%)")
print(f"  - Q < {Q_min}: {q_lower_fail.sum()} ({q_lower_fail.sum()/total*100:.1f}%)")
print(f"  - Q > {Q_max}: {q_upper_fail.sum()} ({q_upper_fail.sum()/total*100:.1f}%)")
print(f"fEO Failure (< {fEO_min} GHz): {feo_fail.sum()} ({feo_fail.sum()/total*100:.1f}%)")
print(f"FSR Failure (< {FSR_min} nm): {fsr_fail.sum()} ({fsr_fail.sum()/total*100:.1f}%)")

# Let's find thresholds to allow 25% of data to be valid
# A simple way is to find quantiles of the data. 
# We want ER >= new_ER_min, Q in new_Q_range, fRC >= new_fRC_min, FSR >= new_FSR_min
# Let's see the distributions
print("\n--- Value distributions (quantiles) ---")
for col in ['ER (dB)', 'Q Factor', 'f_EO (GHz)', 'FSR (nm)', 'Lc (um)']:
    quantiles = df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"\n{col}:")
    print(quantiles)

# Proposing new thresholds iteratively to get at least 25% valid
target_valid = int(total * 0.25)
print(f"\n--- Searching for new thresholds to hit >= {target_valid} valid samples ---")

# Best approach: find thresholds that individually keep ~40-60% of data, so intersection is >25%.
candidates_ER = [5.0, 6.0, 7.0, 8.0, 9.0]
candidates_Q_min = [df['Q Factor'].quantile(0.1), df['Q Factor'].quantile(0.2), 5000, 8000, 9000]
candidates_Q_max = [df['Q Factor'].quantile(0.9), df['Q Factor'].quantile(0.8), 20000, 30000]

best_combination = None
valid_count = 0

for er in candidates_ER:
    for q_L in candidates_Q_min:
        for q_U in candidates_Q_max:
            # maintain feo and fsr constraints untouched if possible, unless they fail heavily
            er_cond = df['ER (dB)'] >= er
            q_cond = (df['Q Factor'] >= q_L) & (df['Q Factor'] <= q_U)
            # if FSR and fEO failed heavily, we would change them. But FSR is usually > 6.4 if R limit is tight.
            # let's assume fEO and FSR are mostly met, if not, relax them too.
            feo_cond = df['f_EO (GHz)'] >= fEO_min
            fsr_cond = df['FSR (nm)'] >= FSR_min
            
            valid_mask = er_cond & q_cond & feo_cond & fsr_cond
            v_count = valid_mask.sum()
            
            if v_count >= target_valid:
                best_combination = (er, q_L, q_U, fEO_min, FSR_min)
                valid_count = v_count
                break
        if best_combination: break
    if best_combination: break

if best_combination:
    print(f"\nFound configuration with {valid_count} valid samples ({valid_count/total*100:.1f}%)!")
    print(f"ER_min: {best_combination[0]}")
    print(f"Q_min: {best_combination[1]:.0f}")
    print(f"Q_max: {best_combination[2]:.0f}")
else:
    print("\nCould not find a combination with simple grid. Let's try aggressive global relaxation.")
    er = df['ER (dB)'].quantile(0.25)
    q_L = df['Q Factor'].quantile(0.1)
    q_U = df['Q Factor'].max() + 1000  # basically no upper bound
    valid_mask = (df['ER (dB)'] >= er) & (df['Q Factor'] >= q_L) & (df['Q Factor'] <= q_U) & (df['f_EO (GHz)'] >= fEO_min ) & (df['FSR (nm)'] >= FSR_min)
    print(f"Aggressive valid samples: {valid_mask.sum()} / {total}")
    print(f"ER_min: {er:.2f}")
    print(f"Q_min: {q_L:.0f}")
    print(f"Q_max: {q_U:.0f}")

