# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:38:38 2021

@author: ashymanskaya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:12:23 2021

@author: ashymanskaya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:54:39 2021

@author: ashymanskaya
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:49:31 2021

@author: ashymanskaya
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from nilearn import plotting
from matplotlib import pyplot as plt
import scipy.io
try:
    from sklearn.covariance import GraphicalLassoCV
except ImportError:
    # for Scitkit-Learn < v0.20.0
    from sklearn.covariance import GraphLassoCV as GraphicalLassoCV
from conpagnon.utils import array_operation
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
import pandas as pd
from conpagnon.utils import pre_preprocessing
from scipy.stats import scoreatpercentile
import importlib
from conpagnon.computing import compute_connectivity_matrices as ccm
from conpagnon.pylearn_mulm import mulm
from nilearn.connectome import ConnectivityMeasure
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy import stats


def plot_matrices(cov, prec, title, labels):
    """Plot covariance and precision matrices, for a given processing. """

    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[list(range(size)), list(range(size))] = 0
    plt.figure()
    # Display covariance matrix
    plotting.plot_matrix(cov, cmap=plotting.cm.bwr,
                         vmin=-1, vmax=1, title="%s / covariance" % title,
                         labels=labels)
    #plt.savefig("covariance_matrix_pat.svg")
    plt.figure()

    # Display precision matrix
    plotting.plot_matrix(prec, cmap=plotting.cm.bwr,
                         vmin=-1, vmax=1, title="%s / precision" % title,
                         labels=labels)
    #plt.savefig("precision_matrix_pat.svg")


def fion_GrLassoCV(time_series):
    gl = GraphicalLassoCV(verbose=2)
    a=gl.fit(np.concatenate(time_series))
    return a


all_time_series_RS1 = []
all_time_series_RS2 = []
my_coords= []
i=0
j=0

df_full = pd.read_excel(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\sample_RS_T1&2.xlsx')
#kick out outliners
df_full=df_full.drop([21,62])
df_group=df_full.Gruppe
df_diag=df_full.MINI_diag

index_pat_healthy=df_group.loc[(df_group==1)&(df_diag==0)].index
index_pat_sick=df_group.loc[(df_group==1)&(df_diag==1)].index

index_HC_healthy=df_group.loc[(df_group==2)&(df_diag==0)].index
index_HC_sick=df_group.loc[(df_group==2)&(df_diag==1)].index

labels=[]
for j in range(1,67):
    if j<10:
        mat = scipy.io.loadmat(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\conn_iuk_RS_T1&2\ROI_Subject00'+str(j)+'_Condition000.mat')
    else:
        mat = scipy.io.loadmat(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\conn_iuk_RS_T1&2\ROI_Subject0'+str(j)+'_Condition000.mat')

    time_series_RS1= np.zeros((246,0))
    time_series_RS2= np.zeros((246,0))

    sources= np.concatenate((np.arange(135,139),np.arange(146,161)))
    sources=[135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154,
       155, 156, 157, 159,158, 160]
    for i in sources:
        time_series_RS1=np.hstack((time_series_RS1,mat['data'][0][i][0:246]))
        time_series_RS2=np.hstack((time_series_RS2,mat['data'][0][i][246:492]))
        
        lbs=mat['names'][0][i].tolist()[0]
        lbs=lbs.split('.')[1].lstrip().split(' ')[0]+'.'+lbs.split('.')[2].lstrip().split(' ')[0]+lbs.split('.')[2].lstrip().split(' ')[1]
        labels=np.hstack((labels,lbs))
        
        
    all_time_series_RS1.append(time_series_RS1)
    all_time_series_RS2.append(time_series_RS2)

for i in sources:        
        my_coords.append( tuple(np.ravel(mat['xyz'][0][i].tolist())))


labels=labels[0:19]

df= pd.read_excel(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\confounds.xlsx')

    
time_series_HC_healthy_RS1 = []
for index in index_HC_healthy:
    time_series_HC_healthy_RS1.append(StandardScaler().fit_transform(all_time_series_RS1[index]))
    
time_series_pat_healthy_RS1 = []
for index in index_pat_healthy:
    time_series_pat_healthy_RS1.append(StandardScaler().fit_transform(all_time_series_RS1[index]))  

time_series_HC_sick_RS1 = []
for index in index_HC_sick:
    time_series_HC_sick_RS1.append(StandardScaler().fit_transform(all_time_series_RS1[index]))
time_series_pat_sick_RS1 = []
for index in index_pat_sick:
    time_series_pat_sick_RS1.append(StandardScaler().fit_transform(all_time_series_RS1[index]))  



n_regions = len(my_coords)

#RS1 sick
gl_HC_sick_RS1 = GraphicalLassoCV(verbose=2)
gl_HC_sick_RS1.fit(np.concatenate(time_series_HC_sick_RS1))

gl_pat_sick_RS1 = GraphicalLassoCV(verbose=2)
gl_pat_sick_RS1.fit(np.concatenate(time_series_pat_sick_RS1))

plotting.plot_connectome(gl_pat_sick_RS1.covariance_,
                         my_coords, edge_threshold='70%',
                         title="Patients, RS1: Covariance",
                         display_mode="lzr")

plotting.plot_connectome(gl_HC_sick_RS1.covariance_,
                         my_coords, edge_threshold='70%',
                         title="HCs, RS1: Covariance",
                         display_mode="lzr")


plt.figure()

plotting.plot_connectome(-gl_pat_sick_RS1.precision_, my_coords,node_color=['blue', 'blue',
       'blue', 'blue',
       'red', 'red',
       'red', 'red', 'red',
       'red', 'red', 'cyan',
       'cyan', 'cyan',
       'cyan', 'yellow',
       'yellow', 'yellow',
       'yellow'],
                         edge_threshold='60%',
                         title="Patients, RS1: Sparse inverse covariance (GraphicalLasso)",
                         display_mode="lzr")
plt.savefig("precision_RS1_sick_pat_final.svg")

plt.figure()

plotting.plot_connectome(-gl_HC_sick_RS1.precision_, my_coords,node_color=['blue', 'blue',
       'blue', 'blue',
       'red', 'red',
       'red', 'red', 'red',
       'red', 'red', 'cyan',
       'cyan', 'cyan',
       'cyan', 'yellow',
       'yellow', 'yellow',
       'yellow'],
                         edge_threshold='60%',
                         title="HCs, RS1: Sparse inverse covariance (GraphicalLasso)",
                         display_mode="lzr")
plt.savefig("precision_RS1_sick_HC_final.svg")

plt.figure()

plot_matrices(gl_pat_sick_RS1.covariance_, -gl_pat_sick_RS1.precision_, "Patients, RS1: GraphicalLasso", labels)
plt.savefig("precision_mat_RS1_sick_pat.svg")

plt.figure()

plot_matrices(gl_HC_sick_RS1.covariance_, -gl_HC_sick_RS1.precision_, "HCs, RS1: GraphicalLasso", labels)
plt.savefig("precision_mat_RS1_sick_HC.svg")


dict_time_series_HC_sick_RS1={str(idx + 1)+'_HC' :time_series_HC_sick_RS1[idx] for idx in range(len(time_series_HC_sick_RS1))}

# bootstrap MC integration
reps = 10000
xb_sick = np.random.choice(list(dict_time_series_HC_sick_RS1.keys()), (reps, len(list(dict_time_series_HC_sick_RS1.keys()))), replace=True)

dict_time_series_pat_sick_RS1={str(idx + 1)+'_pat' :time_series_pat_sick_RS1[idx] for idx in range(len(time_series_pat_sick_RS1))}




bs_dict_time_series_HC_sick_RS1={'bs_run '+str(idx) :{'IDs': xb_sick[idx], 'time series':[dict_time_series_HC_sick_RS1.get(key) for key in xb_sick[idx]],'GraphicalLassoCV':fion_GrLassoCV([dict_time_series_HC_sick_RS1.get(key) for key in xb_sick[idx]])} for idx in range(reps)}

#bs_dict_time_series_pat_RS1={'bs_run '+str(idx) :{'IDs': xb_pat[idx], 'time series':[dict_time_series_pat_RS1.get(key) for key in xb_pat[idx]],'GraphicalLassoCV':fion_GrLassoCV([dict_time_series_pat_RS1.get(key) for key in xb_pat[idx]])} for idx in range(reps)}
#bs_dict_time_series_pat_RS2={'bs_run '+str(idx) :{'IDs': xb_pat_RS2[idx], 'time series':[dict_time_series_pat_RS2.get(key) for key in xb_pat_RS2[idx]],'GraphicalLassoCV':fion_GrLassoCV([dict_time_series_pat_RS2.get(key) for key in xb_pat_RS2[idx]])} for idx in range(reps)}

#dict_patients_RS1={list(bs_dict_time_series_pat_RS1.keys())[idx]:{'GraphicalLassoCV': -bs_dict_time_series_pat_RS1[list(bs_dict_time_series_pat_RS1.keys())[idx]]['GraphicalLassoCV'].precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19)} for idx in range(reps)}
#dict_patients_RS2={list(bs_dict_time_series_pat_RS2.keys())[idx]:{'GraphicalLassoCV': -bs_dict_time_series_pat_RS2[list(bs_dict_time_series_pat_RS2.keys())[idx]]['GraphicalLassoCV'].precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19)} for idx in range(reps)}
dict_patients_sick_RS1 = {'0' :{'GraphicalLassoCV': -fion_GrLassoCV(time_series_pat_sick_RS1).precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19) }}



#dict_patients={list(bs_dict_time_series_pat.keys())[idx]:{'GraphicalLassoCV': -bs_dict_time_series_pat[list(bs_dict_time_series_pat.keys())[idx]]['GraphicalLassoCV'].precision_,'diagonal_mask':np.repeat(False, np.shape(correlation_matrices_pat)[1]), 'masked_array':np.array([np.repeat(False, np.shape(correlation_matrices_pat)[1]),]*np.shape(correlation_matrices_pat)[1])} for idx in range(reps)}
dict_HCs_sick_RS1={list(bs_dict_time_series_HC_sick_RS1.keys())[idx]:{'GraphicalLassoCV': -bs_dict_time_series_HC_sick_RS1[list(bs_dict_time_series_HC_sick_RS1.keys())[idx]]['GraphicalLassoCV'].precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19)} for idx in range(reps)}

lasso_connectivity_matrices_RS1_sick =  {'patients':dict_patients_sick_RS1, 'controls':dict_HCs_sick_RS1}

importlib.reload(pre_preprocessing)
importlib.reload(array_operation)
importlib.reload(ccm)
importlib.reload(mulm)
kinds= ['GraphicalLassoCV' ]

stacked_matrices_RS1_sick = pre_preprocessing.stacked_connectivity_matrices(subjects_connectivity_matrices=
                                                                       lasso_connectivity_matrices_RS1_sick,
                                                                       kinds=kinds)

stacked_matrices=stacked_matrices_RS1_sick.copy()
groups=['patients', 'controls']
X = pre_preprocessing.fisher_transform(symmetric_array=stacked_matrices[groups[0]]['GraphicalLassoCV'])
Y = pre_preprocessing.fisher_transform(symmetric_array=stacked_matrices[groups[1]]['GraphicalLassoCV'])
#X = stacked_matrices[groups[0]]['GraphicalLassoCV']
#Y = stacked_matrices[groups[1]]['GraphicalLassoCV']

X_vectorize = sym_matrix_to_vec(symmetric=X)
Y_vectorize = sym_matrix_to_vec(symmetric=Y)
# Vectorize the corresponding boolean array
vec_X_mask = array_operation.vectorize_boolean_mask(
    symmetric_boolean_mask=stacked_matrices[groups[0]]['masked_array'])
vec_Y_mask = array_operation.vectorize_boolean_mask(
    symmetric_boolean_mask=stacked_matrices[groups[1]]['masked_array'])

# Create a numpy masked array structure for the two sample
X_new = np.ma.array(data=X_vectorize)
Y_new =np.ma.array(data=Y_vectorize)

# Extract the bootstrapped confidence intervals
df_bs = pd.DataFrame(np.array(Y_new))
bs_err_dic = {}

for ind, item in enumerate(df_bs):
    bs_err = scoreatpercentile(df_bs[item], [0.001, 99.999], interpolation_method='fraction', axis=None)
    bs_err_dic[item] = bs_err
    
err_l_ = []
err_u_ = []
err_dict = {}

for ind in range(190):
    err_l =  bs_err_dic[ind][0] -X_new[:,ind].data
    err_u = bs_err_dic[ind][1] - X_new[:,ind].data
    err_l_.append(err_l)
    err_u_.append(err_u)
    if err_l >= 0 and err_u >= 0:
        err_dict[ind] = np.array([err_l, err_u])
    elif err_l <= 0 and err_u <= 0:
        err_dict[ind] = np.array([err_l, err_u])



        
err_dict_lasso=err_dict.copy()
names_abberant_lasso=list(err_dict_lasso.keys())
print(err_dict_lasso)
#err_dict_lasso_RS1=err_dict_lasso.copy()
err_dict_lasso_RS2=err_dict_lasso.copy()

#significant abberations only
X_abb = X_new.copy()
for i in range(190):
    if  i in names_abberant_lasso:  
        print(i)
    else:    
            X_abb[0,i] = 0
X_abb=vec_to_sym_matrix(X_abb)
Y_mean=np.reshape(np.mean(Y_new,axis=0), (1, -1))
Y_abb = Y_mean.copy()
for i in range(190):
    if not  i in names_abberant_lasso:  
        Y_abb[:,i] = 0
Y_abb=vec_to_sym_matrix(Y_abb)
Diff=np.squeeze(X_abb-Y_abb)
    
plt.figure()
plotting.plot_connectome(Diff, my_coords,node_color=['blue', 'blue',
       'blue', 'blue',
       'red', 'red',
       'red', 'red', 'red',
       'red', 'red', 'cyan',
       'cyan', 'cyan',
       'cyan', 'yellow',
       'yellow', 'yellow',
       'yellow'],edge_threshold='90%',
                         title='mean abberation Gauss Lasso CV')
plt.savefig("0_001/difference_precision_RS1_pat-HC_final_sick.svg")

plt.figure()
prec = Diff.copy()  # avoid side effects

# Put zeros on the diagonal, for graph clarity.
size = prec.shape[0]
prec[list(range(size)), list(range(size))] = 0
span = max(abs(prec.min()), abs(prec.max()))
# Display precision matrix
plotting.plot_matrix(Diff, cmap=plotting.cm.bwr,
                      vmin=-1, vmax=1, title="%s / precision" % 'mean abberation VBM',
                      labels=labels)
plt.savefig("0_001/precision_matrix_RS1_pat-HC_final_sick.svg")

a=[37,79,130,182]
#stai_trai=stai_x2
table=df_full[['Gruppe', 'BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome','ChildhoodViolence','Anzahl_GE','Schweregrad','MINI_diag']].copy()

partial_correlation_measure = ConnectivityMeasure(kind='partial correlation')
partial_correlation_matrices = partial_correlation_measure.fit_transform(all_time_series_RS1)

index_sick=sorted(np.concatenate((index_pat_sick,index_HC_sick)))
partial_correlation_measure_vectorize = sym_matrix_to_vec(symmetric=partial_correlation_matrices)[:,a][index_sick]
partial_correlation_measure_vectorize_st=pd.DataFrame(StandardScaler().fit_transform(partial_correlation_measure_vectorize))

df_RS1_for_regression_sick=pd.DataFrame(partial_correlation_measure_vectorize_st)

df_RS1_for_regression_sick=pd.concat([df_RS1_for_regression_sick, table.loc[index_sick].reset_index()], axis=1)
# writer = pd.ExcelWriter('RS1_regression.xlsx', engine='xlsxwriter')
# # Convert the dataframe to an XlsxWriter Excel object.
# df_RS1_for_regression.to_excel(writer, sheet_name='Sheet1')
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()    
writer = pd.ExcelWriter('0_001/RS1_64_regression_sick.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
df_RS1_for_regression_sick.to_excel(writer, sheet_name='Sheet1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()  

raw_RS1_sick=pd.concat([pd.DataFrame(partial_correlation_measure_vectorize).reset_index(drop=True), table.loc[index_sick].reset_index(drop=True)], axis=1)
writer = pd.ExcelWriter('0_001/raw_RS1_regression_64_final_sick.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
raw_RS1_sick.to_excel(writer, sheet_name='Sheet1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()

t_test_HC_pat_sick=dict()
p_value_HC_pat_sick=dict()
table_sick=table.loc[index_sick]
for col in ['BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome']: 
    [t_test_HC_pat_sick[col],p_value_HC_pat_sick[col]]=stats.mannwhitneyu(table_sick.loc[table_sick['Gruppe']==1][col],table_sick.loc[table_sick['Gruppe']==2][col])
    
mtests=multipletests(list(p_value_HC_pat_sick.values()), alpha=0.05, method='hs', is_sorted=False, returnsorted=False)
mtests_names=pd.DataFrame(mtests[0], ['BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome'])
mtests_names[mtests_names[0]==True]


index_healthy=sorted(np.concatenate((index_pat_healthy,index_HC_healthy)))

t_test_HC_pat_healthy=dict()
p_value_HC_pat_healthy=dict()
table_healthy=table.loc[index_healthy]
for col in ['BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome']: 
    [t_test_HC_pat_healthy[col],p_value_HC_pat_healthy[col]]=stats.mannwhitneyu(table_healthy.loc[table_healthy['Gruppe']==1][col],table_healthy.loc[table_healthy['Gruppe']==2][col])
    
mtests=multipletests(list(p_value_HC_pat_healthy.values()), alpha=0.05, method='hs', is_sorted=False, returnsorted=False)
mtests_names=pd.DataFrame(mtests[0], ['BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome'])
mtests_names[mtests_names[0]==True]

t_test_pat_sick_pat_healthy=dict()
p_value_pat_sick_pat_healthy=dict()
for col in ['BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome','ChildhoodViolence','Anzahl_GE','Schweregrad']: 
    [t_test_pat_sick_pat_healthy[col],p_value_pat_sick_pat_healthy[col]]=stats.mannwhitneyu(table.loc[(table['Gruppe']==1)&(table['MINI_diag']==0)][col],table.loc[(table['Gruppe']==1)&(table['MINI_diag']==1)][col])
    
mtests=multipletests(list(p_value_pat_sick_pat_healthy.values()), alpha=0.05, method='hs', is_sorted=False, returnsorted=False)
mtests_names=pd.DataFrame(mtests[0],['BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome','ChildhoodViolence','Anzahl_GE','Schweregrad'])
mtests_names[mtests_names[0]==True]

t_test_HC_sick_HC_healthy=dict()
p_value_HC_sick_HC_healthy=dict()
for col in ['BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome']: 
    [t_test_HC_sick_HC_healthy[col],p_value_HC_sick_HC_healthy[col]]=stats.mannwhitneyu(table.loc[(table['Gruppe']==2)&(table['MINI_diag']==0)][col],table.loc[(table['Gruppe']==2)&(table['MINI_diag']==1)][col])
    
mtests=multipletests(list(p_value_HC_sick_HC_healthy.values()), alpha=0.05, method='hs', is_sorted=False, returnsorted=False)
mtests_names=pd.DataFrame(mtests[0],['BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome'])
mtests_names[mtests_names[0]==True]

