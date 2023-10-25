"""
Created on Tue Jan 24 10:54:59 2023

@author: aubouinb
"""

import pandas as pd


folder_path = './Results_data/'

Table_PID = pd.read_csv(folder_path + 'result_table_PID_induction_500.csv')
Table_NMPC = pd.read_csv(folder_path + 'result_table_MHE_NMPC_induction_500.csv')
Table_MMPC = pd.read_csv(folder_path + 'result_table_MEKF_NMPC_induction_500.csv')


final_table = pd.DataFrame(columns=['Controller', 'TT_mean', 'TT_max', 'BIS_NADIR_mean', 'BIS_NADIR_min',
                                    'ST10_mean', 'ST10_max', 'ST20_mean', 'ST20_max', 'US_mean', 'US_max'])

final_table['Controller'] = ['PID', 'NMPC', 'MMPC']

list_data = [Table_PID, Table_NMPC, Table_MMPC]

final_table['TT_mean'] = [str(df['TT (min)'][0]) + "$\pm$" + str(df['TT (min)'][1]) for df in list_data]
final_table['TT_max'] = [str(df['TT (min)'][3]) for df in list_data]

final_table['BIS_NADIR_mean'] = [str(df['BIS_NADIR'][0]) + "$\pm$" + str(df['BIS_NADIR'][1]) for df in list_data]
final_table['BIS_NADIR_min'] = [str(df['BIS_NADIR'][2]) for df in list_data]

final_table['ST10_mean'] = [str(df['ST10 (min)'][0]) + "$\pm$" + str(df['ST10 (min)'][1]) for df in list_data]
final_table['ST10_max'] = [str(df['ST10 (min)'][3]) for df in list_data]

final_table['ST20_mean'] = [str(df['ST20 (min)'][0]) + "$\pm$" + str(df['ST20 (min)'][1]) for df in list_data]
final_table['ST20_max'] = [str(df['ST20 (min)'][3]) for df in list_data]

final_table['US_mean'] = [str(df['US'][0]) + "$\pm$" + str(df['US'][1]) for df in list_data]
final_table['US_max'] = [str(df['US'][3]) for df in list_data]


styler = final_table.style
styler.hide(axis='index')
styler.format(precision=2)
print(styler.to_latex())
