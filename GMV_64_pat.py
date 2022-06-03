# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:14:41 2021

@author: ashymanskaya
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:00:23 2021

@author: ashymanskaya
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:10:38 2021

@author: ashymanskaya
"""
import os # new
import numpy as np
import nilearn
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats     
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
    plt.figure()

    # Display precision matrix
    plotting.plot_matrix(prec, cmap=plotting.cm.bwr,
                         vmin=-1, vmax=1, title="%s / precision" % title,
                         labels=labels)
    

def fion_GrLassoCV(vbm):
    gl = GraphicalLassoCV(verbose=2)
    a=gl.fit(np.concatenate(vbm))
    return a

df_full = pd.read_excel(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\sample_RS_T1&2.xlsx')
df_full=df_full.drop([21,62])
df_group=df_full.Gruppe
df_diag=df_full.MINI_diag
index_HC=df_group.loc[(df_group==2)].index
index_pat=df_group.loc[(df_group==1)].index
index_pat_healthy=df_group.loc[(df_group==1)&(df_diag==0)].index
index_pat_sick=df_group.loc[(df_group==1)&(df_diag==1)].index

index_HC_healthy=df_group.loc[(df_group==2)&(df_diag==0)].index
index_HC_sick=df_group.loc[(df_group==2)&(df_diag==1)].index

my_coords= []
labels=[]

for j in range(1,67):
    if j<10:
        mat = scipy.io.loadmat(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\conn_iuk_RS_T1&2\ROI_Subject00'+str(j)+'_Condition000.mat')
    else:
        mat = scipy.io.loadmat(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\conn_iuk_RS_T1&2\ROI_Subject0'+str(j)+'_Condition000.mat')


    sources=[135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154,
       155, 156, 157, 159,158, 160]
    for i in sources:

        lbs=mat['names'][0][i].tolist()[0]
        lbs=lbs.split('.')[1].lstrip().split(' ')[0]+'.'+lbs.split('.')[2].lstrip().split(' ')[0]+lbs.split('.')[2].lstrip().split(' ')[1]
        labels=np.hstack((labels,lbs))

for i in sources:        
        my_coords.append( tuple(np.ravel(mat['xyz'][0][i].tolist())))
labels=labels[0:19]

path=r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\structural_data_alex'
list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
all_data= np.empty((66,19), float)
all_data=[]
for j in range(66):
        mat = scipy.io.loadmat(list_subfolders_with_paths[j]+'\\label\\catROI_y-vol_00001.mat')
        data_DMN_SN=(mat['S']['atlas_DMN_SN'][0,0]['data'][0,0]['Vgm'][0,0])
        data_DMN_SN=np.concatenate(data_DMN_SN).astype(None)
        data_DA=(mat['S']['atlas_DA'][0,0]['data'][0,0]['Vgm'][0,0])
        data_DA=np.concatenate(data_DA).astype(None)
        data_FP=(mat['S']['atlas_FP'][0,0]['data'][0,0]['Vgm'][0,0])
        data_FP=np.concatenate(data_FP).astype(None)
        data=np.concatenate((data_DMN_SN,data_DA,data_FP), axis=0)
        all_data.append(data.reshape(1, -1))
all_data= np.squeeze(np.asarray(all_data))

df= pd.read_excel(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\confounds.xlsx')

confounds=[pd.get_dummies(df['NumberOtherPsychopharmaca']).values,pd.get_dummies(df['Mini_diagnose']).values,pd.get_dummies(df['Geschlecht']).values,
                               pd.get_dummies(df['Antidepressiva']).values, np.array(df['Age_centered'])]

confounds_VBM=confounds.copy()
with open(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\structural_data_alex\TIV.txt') as f:
    IV = f.readlines()
TIV = [float(g) for g in IV]
TIV = StandardScaler().fit_transform(np.array(TIV).reshape(-1, 1))
confounds_VBM.append(np.array(TIV))

deconf_all_data = nilearn.signal.clean(all_data,
                                                  confounds=confounds_VBM, standardize=False,standardize_confounds=False)
deconf_all_data=StandardScaler().fit_transform(deconf_all_data)
VBM_pat_sick=deconf_all_data[index_pat_sick,]

#VBM_pat_sick=StandardScaler().fit_transform(deconf_all_data[index_pat_sick,])
#scaler_pat=StandardScaler()
#z_VBM_pat = scaler_pat.fit_transform(VBM_pat[:,0].reshape(-1, 1))
all_VBM_pat_sick=[]
for j in range(0,len(index_pat_sick)):
        time_series_sick=VBM_pat_sick[j,].reshape(1, -1)
        all_VBM_pat_sick.append(time_series_sick)
VBM_HC_sick=deconf_all_data[index_HC_sick,]

#VBM_HC_sick=StandardScaler().fit_transform(deconf_all_data[index_HC_sick,])
#scaler_HC=StandardScaler()
#z_VBM_HC = scaler_HC.fit_transform(VBM_HC[:,0].reshape(-1, 1))
all_VBM_HC_sick=[]
for j in range(0,len(index_HC_sick)):
        time_series_sick=VBM_HC_sick[j,].reshape(1, -1)
        all_VBM_HC_sick.append(time_series_sick)
p_val= []       
for i in range(19):
    t,p=stats.mannwhitneyu(VBM_HC_sick[:,i], VBM_pat_sick[:,i])
    p_val.append(p)
    
gl_HC_sick = GraphicalLassoCV(verbose=2)
gl_HC_sick.fit(np.concatenate(all_VBM_HC_sick))

gl_pat_sick = GraphicalLassoCV(verbose=2)
gl_pat_sick.fit(np.concatenate(all_VBM_pat_sick))

plotting.plot_connectome(gl_pat_sick.covariance_,
                         my_coords, edge_threshold='50%',
                         title="Patients: Covariance",
                         display_mode="lzr")
plotting.plot_connectome(gl_HC_sick.covariance_,
                         my_coords, edge_threshold='50%',
                         title="Hcs: Covariance",
                         display_mode="lzr")
plt.figure()

plotting.plot_connectome(-gl_pat_sick.precision_, my_coords,node_color=['blue', 'blue',
       'blue', 'blue',
       'red', 'red',
       'red', 'red', 'red',
       'red', 'red', 'cyan',
       'cyan', 'cyan',
       'cyan', 'yellow',
       'yellow', 'yellow',
       'yellow'],
                         edge_threshold='50%',
                         title="Patients: GraphicalLassoCV)",
                         display_mode="lzr")
plt.savefig("0_001/precision_GMV_pat_sick_final.svg")

plt.figure()

plotting.plot_connectome(-gl_HC_sick.precision_, my_coords,node_color=['blue', 'blue',
       'blue', 'blue',
       'red', 'red',
       'red', 'red', 'red',
       'red', 'red', 'cyan',
       'cyan', 'cyan',
       'cyan', 'yellow',
       'yellow', 'yellow',
       'yellow'],
                         edge_threshold='50%',
                         title="HCs: GraphicalLassoCV",
                         display_mode="lzr")
plt.savefig("0_001/precision_GMV_HC_sick_final.svg")

plt.figure()

plot_matrices(gl_pat_sick.covariance_, -gl_pat_sick.precision_, "Patients: GraphicalLassoCV", labels)
plt.savefig("0_001/precision_mat_GMV_pat_sick_final.svg")

plt.figure()

plot_matrices(gl_HC_sick.covariance_, -gl_HC_sick.precision_, "HCs: GraphicalLassoCV", labels)
plt.savefig("0_001/precision_mat_GMV_HC_sick_final.svg")





dict_VBM_HC_sick={str(idx + 1)+'_HC' :all_VBM_HC_sick[idx] for idx in range(len(all_VBM_HC_sick))}

# bootstrap MC integration
reps = 10000
xb_sick = np.random.choice(list(dict_VBM_HC_sick.keys()), (reps, len(list(dict_VBM_HC_sick.keys()))), replace=True)

dict_VBM_pat_sick={str(idx + 1)+'_HC' :all_VBM_pat_sick[idx] for idx in range(len(all_VBM_pat_sick))}
xb_pat_sick = np.random.choice(list(dict_VBM_pat_sick.keys()), (reps, len(list(dict_VBM_pat_sick.keys()))), replace=True)


bs_dict_VBM_HC_sick={'bs_run '+str(idx) :{'IDs': xb_sick[idx], 'time series':[dict_VBM_HC_sick.get(key) for key in xb_sick[idx]],'GraphicalLassoCV':fion_GrLassoCV([dict_VBM_HC_sick.get(key) for key in xb_sick[idx]])} for idx in range(reps)}
dict_patients_sick = {'0' :{'GraphicalLassoCV': -fion_GrLassoCV(all_VBM_pat_sick).precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19) }}
#bs_dict_VBM_pat={'bs_run '+str(idx) :{'IDs': xb_pat[idx], 'time series':[dict_VBM_pat.get(key) for key in xb_pat[idx]],'GraphicalLassoCV':fion_GrLassoCV([dict_VBM_pat.get(key) for key in xb_pat[idx]])} for idx in range(reps)}
#dict_patients={list(bs_dict_VBM_pat.keys())[idx]:{'GraphicalLassoCV': -bs_dict_VBM_pat[list(bs_dict_VBM_pat.keys())[idx]]['GraphicalLassoCV'].precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19)} for idx in range(reps)}
dict_HCs_GMV_sick={list(bs_dict_VBM_HC_sick.keys())[idx]:{'GraphicalLassoCV': -bs_dict_VBM_HC_sick[list(bs_dict_VBM_HC_sick.keys())[idx]]['GraphicalLassoCV'].precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19)} for idx in range(reps)}

bs_dict_VBM_HC_GMV_sick=bs_dict_VBM_HC_sick.copy()
dict_patients_GMV_sick=dict_patients_sick.copy()

lasso_connectivity_matrices_sick =  {'patients':dict_patients_GMV_sick, 'controls':dict_HCs_GMV_sick}
kinds= ['GraphicalLassoCV' ]

stacked_matrices_sick = pre_preprocessing.stacked_connectivity_matrices(subjects_connectivity_matrices=
                                                                       lasso_connectivity_matrices_sick,
                                                                       kinds=kinds)

stacked_matrices_GMV_10k=stacked_matrices_sick.copy()

# X = pre_preprocessing.fisher_transform(symmetric_array=stacked_matrices[groups[0]]['partial_correlation'])
# Y = pre_preprocessing.fisher_transform(symmetric_array=stacked_matrices[groups[1]]['partial_correlation'])
# X = stacked_matrices[groups[0]]['GraphicalLassoCV']
# Y = stacked_matrices[groups[1]]['GraphicalLassoCV']
groups=['patients', 'controls']
#X = pre_preprocessing.fisher_transform(symmetric_array=stacked_matrices[groups[0]]['GraphicalLassoCV'])
#Y = pre_preprocessing.fisher_transform(symmetric_array=stacked_matrices[groups[1]]['GraphicalLassoCV'])

X = stacked_matrices_GMV_10k[groups[0]]['GraphicalLassoCV']
Y = stacked_matrices_GMV_10k[groups[1]]['GraphicalLassoCV']

X_vectorize = sym_matrix_to_vec(symmetric=X)
Y_vectorize = sym_matrix_to_vec(symmetric=Y)
# Vectorize the corresponding boolean array
vec_X_mask = array_operation.vectorize_boolean_mask(
    symmetric_boolean_mask=stacked_matrices_sick[groups[0]]['masked_array'])
vec_Y_mask = array_operation.vectorize_boolean_mask(
    symmetric_boolean_mask=stacked_matrices_sick[groups[1]]['masked_array'])

# Create a numpy masked array structure for the two sample
X_new = np.ma.array(data=X_vectorize)
Y_new = np.ma.array(data=Y_vectorize)


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
    err_l = bs_err_dic[ind][0] -X_new[:,ind].data 
    err_u = bs_err_dic[ind][1] - X_new[:,ind].data
    err_l_.append(err_l)
    err_u_.append(err_u)
    if err_l >= 0 and err_u >= 0:
        err_dict[ind] = np.array([err_l, err_u])
    elif err_l <= 0 and err_u <= 0:
        err_dict[ind] = np.array([err_l, err_u])
        
err_dict_lasso=err_dict.copy()
names_abberant_lasso=list(err_dict_lasso.keys())
conn_GMV_sick=names_abberant_lasso.copy()

print(err_dict_lasso)
err_dict_lasso_GMV=err_dict_lasso.copy()

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
plt.savefig("0_001/difference_precision_VBM_GMV_pat-HC_sick_final.svg")
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
plt.savefig("0_001/precision_matrix_VBM_GMV_pat-HC_sick_final.svg")


a=[3,6,7,8,11,12,13,14,15,16,17,18]
#stai_trai=stai_x2
table=df_full[['Gruppe', 'BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome','ChildhoodViolence','Anzahl_GE','Schweregrad','MINI_diag']].copy()
#table.fillna(table.median(), inplace=True)

index_sick=sorted(np.concatenate((index_pat_sick,index_HC_sick)))

df_pat_GMV=pd.DataFrame(StandardScaler().fit_transform(pd.DataFrame(all_data[:,a][index_sick])))
df_GMV=pd.concat([df_pat_GMV, table.loc[index_sick].reset_index(drop=True)], axis=1)

writer = pd.ExcelWriter('0_001/GMV_regression_64_sick_final.xlsx', engine='xlsxwriter')
# Convert the datafsrame to an XlsxWriter Excel object.
df_GMV.to_excel(writer, sheet_name='Sheet1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()  

raw_GMV_sick=pd.concat([pd.DataFrame(deconf_all_data[:,a][index_sick]).reset_index(drop=True), table.loc[index_sick].reset_index(drop=True)], axis=1)
writer = pd.ExcelWriter('0_001/raw_GMV_regression_64_final_sick.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
raw_GMV_sick.to_excel(writer, sheet_name='Sheet1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()