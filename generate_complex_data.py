import pandas as pd
import numpy as np
import os

def create_complex_dataset():
    np.random.seed(42)
    
    n_patients = 50
    timepoints = ['Baseline', 'Month3', 'Month6', 'Month12']
    
    data = []
    
    for pt in range(1, n_patients + 1):
        group = np.random.choice(['DrugA', 'Placebo'])
        age = np.random.randint(30, 75)
        gender = np.random.choice(['M', 'F'])
        
        # Base BP for the patient
        base_bp = np.random.normal(140, 15) if group == 'Placebo' else np.random.normal(145, 12)
        
        for t_idx, t in enumerate(timepoints):
            # Drug A lowers BP over time; Placebo does not
            if group == 'DrugA':
                bp_drop = t_idx * 5 + np.random.normal(0, 3)
            else:
                bp_drop = np.random.normal(0, 5) # random fluctuation
                
            current_bp = max(90, base_bp - bp_drop)
            
            data.append({
                'PatientID': pt,
                'TreatmentGroup': group,
                'Timepoint': t,
                'Age': age,
                'Gender': gender,
                'BloodPressure': round(current_bp, 1)
            })
            
    df = pd.DataFrame(data)
    
    out_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'clinical_trial.csv')
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} with {len(df)} rows.")

if __name__ == '__main__':
    create_complex_dataset()
