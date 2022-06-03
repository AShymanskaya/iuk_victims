# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:14:50 2021

@author: ashymanskaya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:00:31 2021

@author: ashymanskaya
"""

import os # new
import numpy as np
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
# Display the correlation matrix
from conpagnon.utils import pre_preprocessing
import nibabel as nib
from  nilearn.input_data import NiftiMapsMasker
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

data=list()
for i in list_subfolders_with_paths:
    dt = (str(i)+'\mri\mwp1y-vol_00001.nii')
    data.append(dt)
    
vbm_path = np.array(data)
networks = nib.load(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\structural_data_alex\networks.nii')

df= pd.read_excel(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\confounds.xlsx')

confounds=[pd.get_dummies(df['NumberOtherPsychopharmaca']).values,pd.get_dummies(df['Mini_diagnose']).values,pd.get_dummies(df['Geschlecht']).values,
                               pd.get_dummies(df['Antidepressiva']).values, np.array(df['Age_centered'])]


masker = NiftiMapsMasker(networks,  resampling_target='maps',smoothing_fwhm=2,standardize=True, standardize_confounds=False)
masker.fit()
confounds_VBM=confounds.copy()
with open(r'C:\Users\ashymanskaya\Desktop\Lisa_iuK\_CONN_Teresa\structural_data_alex\TIV.txt') as f:
    IV = f.readlines()
TIV = [float(g) for g in IV]
TIV = StandardScaler().fit_transform(np.array(TIV).reshape(-1, 1))
confounds_VBM.append(np.array(TIV))
masked_data = masker.transform(vbm_path, confounds=confounds_VBM)

masked_data=StandardScaler().fit_transform(masked_data)
VBM_pat=masked_data[index_pat,0:4]
VBM_pat=np.append(VBM_pat,masked_data[index_pat,11:26],axis=1 )
all_VBM_pat=[]
for j in range(0,32):
        time_series=VBM_pat[j,].reshape(1, -1)
        all_VBM_pat.append(time_series)
VBM_HC=masked_data[index_HC,0:4]
VBM_HC=np.append(VBM_HC,masked_data[index_HC,11:26],axis=1 )
all_VBM_HC=[]
for j in range(0,32):
        time_series=VBM_HC[j,].reshape(1, -1)
        all_VBM_HC.append(time_series)
p_val= []  
for i in range(19):
    t,p=stats.mannwhitneyu(VBM_HC[:,i], VBM_pat[:,i])
    p_val.append(p)
    
gl_HC = GraphicalLassoCV(verbose=2)
gl_HC.fit(np.concatenate(all_VBM_HC))

gl_pat = GraphicalLassoCV(verbose=2)
gl_pat.fit(np.concatenate(all_VBM_pat))


plotting.plot_connectome(gl_pat.covariance_,
                         my_coords, edge_threshold='50%',
                         title="Patients: Covariance",
                         display_mode="lzr")
plotting.plot_connectome(gl_HC.covariance_,
                         my_coords, edge_threshold='50%',
                         title="Hcs: Covariance",
                         display_mode="lzr")
plt.figure()
plotting.plot_connectome(-gl_pat.precision_, my_coords,node_color=['blue', 'blue',
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
plt.savefig("precision_GMD_pat_final.svg")

plt.figure()
plotting.plot_connectome(-gl_HC.precision_, my_coords,node_color=['blue', 'blue',
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
plt.savefig("precision_GMD_HC_final.svg")

plt.figure()
plot_matrices(gl_pat.covariance_, -gl_pat.precision_, "Patients: GraphicalLassoCV", labels)
plt.savefig("precision_mat_GMD_pat_final.svg")

plt.figure()
plot_matrices(gl_HC.covariance_, -gl_HC.precision_, "HCs: GraphicalLassoCV", labels)
plt.savefig("precision_mat_GMD_HC_final.svg")





dict_VBM_HC={str(idx + 1)+'_HC' :all_VBM_HC[idx] for idx in range(len(all_VBM_HC))}

# bootstrap MC integration
reps = 10000
xb = np.random.choice(list(dict_VBM_HC.keys()), (reps, len(list(dict_VBM_HC.keys()))), replace=True)

dict_VBM_pat={str(idx + 1)+'_HC' :all_VBM_pat[idx] for idx in range(len(all_VBM_pat))}
xb_pat = np.random.choice(list(dict_VBM_pat.keys()), (reps, len(list(dict_VBM_pat.keys()))), replace=True)



bs_dict_VBM_HC={'bs_run '+str(idx) :{'IDs': xb[idx], 'time series':[dict_VBM_HC.get(key) for key in xb[idx]],'GraphicalLassoCV':fion_GrLassoCV([dict_VBM_HC.get(key) for key in xb[idx]])} for idx in range(reps)}
dict_patients = {'0' :{'GraphicalLassoCV': -fion_GrLassoCV(all_VBM_pat).precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19) }}
#bs_dict_VBM_pat={'bs_run '+str(idx) :{'IDs': xb_pat[idx], 'time series':[dict_VBM_pat.get(key) for key in xb_pat[idx]],'GraphicalLassoCV':fion_GrLassoCV([dict_VBM_pat.get(key) for key in xb_pat[idx]])} for idx in range(reps)}
#dict_patients={list(bs_dict_VBM_pat.keys())[idx]:{'GraphicalLassoCV': -bs_dict_VBM_pat[list(bs_dict_VBM_pat.keys())[idx]]['GraphicalLassoCV'].precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19)} for idx in range(reps)}
dict_HCs={list(bs_dict_VBM_HC.keys())[idx]:{'GraphicalLassoCV': -bs_dict_VBM_HC[list(bs_dict_VBM_HC.keys())[idx]]['GraphicalLassoCV'].precision_,'diagonal_mask':np.repeat(False, 19), 'masked_array':np.array([np.repeat(False, 19),]*19)} for idx in range(reps)}

bs_dict_VBM_HC_GMD=bs_dict_VBM_HC.copy()
dict_patients_GMD=dict_patients.copy()

lasso_connectivity_matrices =  {'patients':dict_patients_GMD, 'controls':dict_HCs}
kinds= ['GraphicalLassoCV' ]

stacked_matrices = pre_preprocessing.stacked_connectivity_matrices(subjects_connectivity_matrices=
                                                                       lasso_connectivity_matrices,
                                                                       kinds=kinds)

stacked_matrices_GMD_10k=stacked_matrices
groups=['patients', 'controls']

X = stacked_matrices_GMD_10k[groups[0]]['GraphicalLassoCV']
Y = stacked_matrices_GMD_10k[groups[1]]['GraphicalLassoCV']

X_vectorize = sym_matrix_to_vec(symmetric=X)
Y_vectorize = sym_matrix_to_vec(symmetric=Y)
# Vectorize the corresponding boolean array
vec_X_mask = array_operation.vectorize_boolean_mask(
    symmetric_boolean_mask=stacked_matrices[groups[0]]['masked_array'])
vec_Y_mask = array_operation.vectorize_boolean_mask(
    symmetric_boolean_mask=stacked_matrices[groups[1]]['masked_array'])

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
conn_GMD=names_abberant_lasso.copy()


print(err_dict_lasso)
err_dict_lasso_GMD=err_dict_lasso.copy()
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
plt.savefig("0_001/difference_precision_VBM_GVD_pat-HC_final.svg")
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
plt.savefig("0_001/precision_matrix_VBM_GVD_pat-HC_final.svg")

a=[4,6,7,13,14,15,17]
#6,8,15
table=df_full[['Gruppe', 'BDI_Final', 'STAI_X2', 'Stress_Gesamt','SCI_Stressymptome','ChildhoodViolence','Anzahl_GE','Schweregrad','MINI_diag']].copy()
index=np.array(range(4))

appended = np.concatenate([index, np.array(range(11,26))])
data_GMD=masked_data[:,appended].copy()

df_pat_GMD=pd.DataFrame(StandardScaler().fit_transform(pd.DataFrame(data_GMD[:,a]).drop([21,62])))
df_GMD=pd.concat([df_pat_GMD, table.reset_index(drop=True)], axis=1)

writer = pd.ExcelWriter('0_001/GMD_regression_64_final.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
df_GMD.to_excel(writer, sheet_name='Sheet1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()      

raw_GMD=pd.concat([pd.DataFrame(data_GMD[:,a]).drop([21,62]).reset_index(drop=True), table.reset_index(drop=True)], axis=1)
writer = pd.ExcelWriter('0_001/raw_GMD_regression_64_final.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
raw_GMD.to_excel(writer, sheet_name='Sheet1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()   
   