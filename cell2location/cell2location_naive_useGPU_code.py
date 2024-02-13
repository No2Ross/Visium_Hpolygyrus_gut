import numpy as np
import pandas as pd
import random
import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import scipy 
from sklearn import cluster
from sklearn import neighbors
import cell2location

import warnings
warnings.filterwarnings("ignore")

kwargs = {"accelerator" : "cpu"}

#Load in visium data
day7_visium = sc.read_h5ad("/mnt/data/project0001/ross/marta_visium/cell2location_naive1/input_files/naive_merged_sections.h5ad")

#Load in reference data
intestine_immune = sc.read_h5ad("/mnt/data/project0001/ross/marta_visium/cell2location_naive1/input_files/xu_habert_merge_obj.h5ad")

intestine_immune = intestine_immune.raw.to_adata()

#The Raw data slot 'var' column names is "_index" which scanpy doesn't like when it comes to saving the object.
#Changing it to be 'features'

intestine_immune.var = intestine_immune.var.rename(columns = {"_index":"features"})


from cell2location.utils.filtering import filter_genes
selected = filter_genes(intestine_immune, cell_count_cutoff=5, cell_percentage_cutoff2=0.08, nonz_mean_cutoff=1.4)



cell2location.models.RegressionModel.setup_anndata(adata=intestine_immune,
                        # 10X reaction / sample / batch
                        batch_key='sample',
                        # cell type, covariate used for constructing signatures
                        labels_key='simpleCellType',
                        # multiplicative technical effects (platform, 3' vs 5', donor effect)
                        categorical_covariate_keys=['condition']
                       )
                  
                  
                  
# create the regression model
from cell2location.models import RegressionModel
mod = RegressionModel(intestine_immune)

# view anndata_setup as a sanity check
mod.view_anndata_setup()                       

mod.train(max_epochs=250, use_gpu=True, accelerator = 'cuda', fast_dev_run = False)




mod.plot_history(20)


plt.savefig("/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/history_train_plot_scRNA.pdf",
                bbox_inches='tight')
plt.close()


# In this section, we export the estimated cell abundance (summary of the posterior distribution).
intestine_immune = mod.export_posterior(
    intestine_immune, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
)

# Save model
mod.save('/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/scRNA_seq_model', overwrite=True)

# Save anndata object with results
adata_file = "/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/sc_GPU.h5ad"
intestine_immune.write(adata_file)
adata_file

mod.plot_QC()

plt.savefig("/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/accuracy_plot_scRNA.pdf",
                bbox_inches='tight')
plt.close()


adata_file = "/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/sc_GPU.h5ad"
intestine_immune = sc.read_h5ad(adata_file)
mod = cell2location.models.RegressionModel.load(f"/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/scRNA_seq_model", intestine_immune)

intestine_immune.var_names = intestine_immune.var['features']

# export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in intestine_immune.varm.keys():
    inf_aver = intestine_immune.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in intestine_immune.uns['mod']['factor_names']]].copy()
else:
    inf_aver = intestine_immune.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in intestine_immune.uns['mod']['factor_names']]].copy()
inf_aver.columns = intestine_immune.uns['mod']['factor_names']
inf_aver.iloc[0:5, 0:5]



#Spatial section

# find shared genes and subset both anndata and reference signatures
intersect = np.intersect1d(day7_visium.var_names, inf_aver.index)
day7_visium = day7_visium[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

# prepare anndata for cell2location model
cell2location.models.Cell2location.setup_anndata(adata=day7_visium)


# create and train the model
mod = cell2location.models.Cell2location(
    day7_visium, cell_state_df=inf_aver,
    # the expected average cell abundance: tissue-dependent
    # hyper-prior which can be estimated from paired histology:
    N_cells_per_location=50,
    # hyperparameter controlling normalisation of
    # within-experiment variation in RNA detection:
    detection_alpha=20
)
mod.view_anndata_setup()



mod.train(max_epochs=30000,
          # train using full data (batch_size=None)
          batch_size=None,
          # use all data points in training because
          # we need to estimate cell abundance at all locations
          train_size=1,
          use_gpu=True, 
          accelerator = 'cuda',
	  fast_dev_run = False
         )

# plot ELBO loss history during training, removing first 100 epochs from the plot
mod.plot_history(1000)
plt.legend(labels=['full data training']);


# In this section, we export the estimated cell abundance (summary of the posterior distribution).
day7_visium = mod.export_posterior(
    day7_visium, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
)

# Save model
mod.save("/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/spatial_model", overwrite=True)

#mod = cell2location.models.Cell2location.load("/mnt/data/project0001/ross/marta_visium/cell2location/object_files/spatial_model", day7_visium)

mod.plot_QC()

plt.savefig("/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/accuracy_plot_spatial.pdf",
                bbox_inches='tight')
plt.close()

# Save anndata object with results
adata_file = "/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/sp_GPU.h5ad"
day7_visium.write(adata_file)
adata_file


adata_file = "/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/sp_GPU.h5ad"
day7_visium = sc.read_h5ad(adata_file)
mod = cell2location.models.Cell2location.load("/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/spatial_model", day7_visium)


# add 5% quantile, representing confident cell abundance, 'at least this amount is present',
# to adata.obs with nice names for plotting
day7_visium.obs[day7_visium.uns['mod']['factor_names']] = day7_visium.obsm['q05_cell_abundance_w_sf']

adata_file = "/mnt/data/project0001/ross/marta_visium/cell2location_naive1/object_files/sp_c2c_annotated_GPU.h5ad"
day7_visium.write(adata_file)
adata_file


