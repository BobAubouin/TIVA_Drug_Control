import json

import pandas as pd
import matplotlib.pyplot as plt

from python_anesthesia_simulator import metrics


# study to load
# ['PID_tot', 'MEKF_NMPC_tot', 'MHE_NMPC_tot_2']  # , 'MHE_NMPC_3', 'MEKF_NMPC_1']
# ['PID_tot_IAE', 'MEKF_NMPC_IAE', 'MHE_NMPC_tot_IAE']
study_name_list = ['PID_mixt', 'MEKF_mixt', 'MHE_mixt']


results = []
for study_name in study_name_list:
    # load the study
    with open(f'data/logs/{study_name}.json', 'r') as f:
        dict = json.load(f)

    # load the results
    filename = f'data/signals/{dict["filename"]}'
    results.append(pd.read_csv(filename))


df_metrics = None

MAINTENANCE_TIME = 599

for i, result in enumerate(results):
    for caseid, df_case in result.groupby('caseid'):
        df_line = metrics.compute_control_metrics(df_case.Time.values,
                                                  df_case.BIS.values,
                                                  phase='total',
                                                  start_step=600,
                                                  end_step=900)
        df_line['study'] = study_name_list[i]
        df_line['IAE_i'] = metrics.intergal_absolut_error(df_case[df_case.Time <= MAINTENANCE_TIME].Time.values,
                                                          df_case[df_case.Time <= MAINTENANCE_TIME].BIS.values)

        df_line['IAE_m'] = metrics.intergal_absolut_error(df_case[df_case.Time > MAINTENANCE_TIME].Time.values,
                                                          df_case[df_case.Time > MAINTENANCE_TIME].BIS.values)
        if df_metrics is None:
            df_metrics = df_line
        else:
            df_metrics = pd.concat((df_metrics, df_line), axis=0)

# Compute mean_std, min, and max for each column grouped by 'study'
df_stats = df_metrics.groupby('study').agg(['mean', 'std', 'min', 'max'])

for metric in df_metrics.columns:
    if metric == 'study':
        continue
    if 'IAE' in metric:
        df_stats[metric, 'mean'] = df_stats[metric, 'mean'].astype(int).astype(
            str) + ' ± ' + df_stats[metric, 'std'].astype(int).astype(str)
        df_stats[metric, 'min'] = df_stats[metric, 'min'].astype(int).round(round_number).astype(str)
        df_stats[metric, 'max'] = df_stats[metric, 'max'].astype(int).round(round_number).astype(str)
    else:
        round_number = 2
        df_stats[metric, 'mean'] = df_stats[metric, 'mean'].astype(float).round(round_number).astype(
            str) + ' ± ' + df_stats[metric, 'std'].astype(float).round(round_number).astype(str)
        df_stats[metric, 'min'] = df_stats[metric, 'min'].astype(float).round(round_number).astype(str)
        df_stats[metric, 'max'] = df_stats[metric, 'max'].astype(float).round(round_number).astype(str)
    df_stats = df_stats.drop(columns=[(metric, 'std')])

# remove specific columns
df_stats.drop(columns=[('IAE_i', 'min'),
                       ('IAE_m', 'min'),
                       ('TT', 'min'),
                       ('BIS_NADIR', 'max'),
                       ('ST10', 'min'),
                       ('ST20', 'min'),
                       ('TTp', 'min'),
                       ('BIS_NADIRp', 'max'),
                       ('TTn', 'min'),
                       ('BIS_NADIRn', 'max'),
                       ('US', 'mean'),
                       ('US', 'min'),
                       ('US', 'max')], inplace=True)

print(df_stats)

df_stats_induction = df_stats[['IAE_i', 'TT', 'BIS_NADIR', 'ST10', 'ST20']]
df_stats_maintenance = df_stats[['IAE_m', 'TTp', 'BIS_NADIRp', 'TTn', 'BIS_NADIRn']]

# export induction to latex table
styler = df_stats_induction.style
filename = f'./outputs/inducion_{"".join(study_name_list)}.tex'
styler.to_latex(filename, hrules=True, column_format='l|'+'cc|'*5)

# in the latex file replace every '_' by '\_'
# in the latex file replace every '±' by '$\pm$'

with open(filename, 'r') as file:
    filedata = file.read()
    filedata = filedata.replace('_', '\_')
    filedata = filedata.replace('±', '$\pm$')
with open(filename, 'w') as file:
    file.write(filedata)

# export induction to latex table
styler = df_stats_maintenance.style
filename = f'./outputs/maintenance_{"".join(study_name_list)}.tex'
styler.to_latex(filename, hrules=True, column_format='l|'+'cc|'*5)

# in the latex file replace every '_' by '\_'
# in the latex file replace every '±' by '$\pm$'

with open(filename, 'r') as file:
    filedata = file.read()
    filedata = filedata.replace('_', '\_')
    filedata = filedata.replace('±', '$\pm$')
with open(filename, 'w') as file:
    file.write(filedata)
