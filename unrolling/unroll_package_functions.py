from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import (
    Any,
    Union,  # noqa: F401
)
import random
import numpy as np
import pandas as pd
import anndata as ad
from scanpy import logging as logg
from scipy.sparse import csr_matrix
from sklearn import neighbors
import squidpy as sq
import scanpy as sc
import scipy
from pyslingshot import Slingshot

import matplotlib.pyplot as plt


from squidpy._constants._pkg_constants import Key
from squidpy.datasets._utils import PathLike
from squidpy.read._utils import _load_image, _read_counts

import igraph as ig
import leidenalg as la
from natsort import natsorted
from sklearn import neighbors
from sklearn import preprocessing


#The package requires that:
#- There cannot be any gaps between connected bits of basement or upper. Even if a spot is unlabelled, use it to connect bits of upper or basement together. The unlabelled spots get removed, so they will not be in your final analysis 

#- A section of upper or basement can only be utilised if it contains more than 10 spots; otherwise slingshot does not function as there are too few points.

#- Spots that are unlabelled in the region annotation can still be left in if they are used for connect bits of upper and basement. These can be removed after the unrolling has been run

#- Choosing the centre spot is incredibly important if you don't have a perfect roll. If bits of the roll overlap or take odd angles, it can lead to uppers being assigned
#to the wrong basements or vice versa



#Perhaps I could match and then merge the upper and basement in one step then do pseudotime on the merged object, without having to do it independently. I have a sneaking suspicion that i tried this and
#it didn't work, but perhaps I just haven't tried it


def get_igraph_from_adjacency(adjacency, directed=None):
        """Get igraph graph from adjacency matrix."""
        import igraph as ig

        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = ig.Graph(directed=directed)
        g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es["weight"] = weights
        except KeyError:
            pass
        if g.vcount() != adjacency.shape[0]:
            logg.warning(
                f"The constructed graph has only {g.vcount()} nodes. "
                "Your adjacency matrix contained redundant nodes."
            )
        return g

def get_sparse_from_igraph(graph, weight_attr=None):
    from scipy.sparse import csr_matrix

    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        weights = graph.es[weight_attr]
    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)
    shape = graph.vcount()
    shape = (shape, shape)
    if len(edges) > 0:
        return csr_matrix((weights, zip(*edges)), shape=shape)
    else:
        return csr_matrix(shape)
    
def leiden_visium(adata, resolution, obsm_space = 'spatial'):
    import igraph as ig
    import leidenalg as la
    from natsort import natsorted
    from sklearn import neighbors


    def get_igraph_from_adjacency(adjacency, directed=None):
        """Get igraph graph from adjacency matrix."""
        import igraph as ig

        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = ig.Graph(directed=directed)
        g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es["weight"] = weights
        except KeyError:
            pass
        if g.vcount() != adjacency.shape[0]:
            logg.warning(
                f"The constructed graph has only {g.vcount()} nodes. "
                "Your adjacency matrix contained redundant nodes."
            )
        return g

    df = pd.DataFrame(adata.obsm[obsm_space])

    knn_graph = neighbors.radius_neighbors_graph(df, radius=33)

    G = get_igraph_from_adjacency(knn_graph.toarray())
    
    partition_kwargs = dict()

    partition_kwargs["resolution_parameter"] = resolution

    partition = la.find_partition(G, la.RBConfigurationVertexPartition, **partition_kwargs)

    groups = np.array(partition.membership)

    adata.obs["leiden_cluster"] = pd.Categorical(
            values=groups.astype("U"),
            categories=natsorted(map(str, np.unique(groups))),
        )
    
    return(adata)



#####
###When loading in the tissue_positions file, it reads from the 2nd (i.e. line 1) in the original code, and not from line 0 like it should

__all__ = ["visium", "vizgen", "nanostring"]


def visium(
    path: PathLike,
    *,
    counts_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool = True,
    source_image_path: PathLike | None = None,
    **kwargs: Any,
) -> AnnData:
    """
    Read *10x Genomics* Visium formatted dataset.

    In addition to reading the regular *Visium* output, it looks for the *spatial* directory and loads the images,
    spatial coordinates and scale factors.

    .. seealso::

        - `Space Ranger output <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.
        - :func:`squidpy.pl.spatial_scatter` on how to plot spatial data.

    Parameters
    ----------
    path
        Path to the root directory containing *Visium* files.
    counts_file
        Which file in the passed directory to use as the count file. Typically either *filtered_feature_bc_matrix.h5* or
        *raw_feature_bc_matrix.h5*.
    library_id
        Identifier for the *Visium* library. Useful when concatenating multiple :class:`anndata.AnnData` objects.
    kwargs
        Keyword arguments for :func:`scanpy.read_10x_h5`, :func:`anndata.read_mtx` or :func:`read_text`.

    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` - spatial spot coordinates.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['images']`` - *hires* and *lowres* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['scalefactors']`` - scale factors for the spots.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['metadata']`` - various metadata.
    """  # noqa: E501
    path = Path(path)
    adata, library_id = _read_counts(path, count_file=counts_file, library_id=library_id, **kwargs)

    if not load_images:
        return adata

    adata.uns[Key.uns.spatial][library_id][Key.uns.image_key] = {
        res: _load_image(path / f"{Key.uns.spatial}/tissue_{res}_image.png") for res in ["hires", "lowres"]
    }
    adata.uns[Key.uns.spatial][library_id]["scalefactors"] = json.loads(
        (path / f"{Key.uns.spatial}/scalefactors_json.json").read_bytes()
    )

    tissue_positions_file = (
        path / "spatial/tissue_positions.csv"
        if (path / "spatial/tissue_positions.csv").exists()
        else path / "spatial/tissue_positions_list.csv"
    )

    coords = pd.read_csv(
        tissue_positions_file,
        header=0 if tissue_positions_file.name == "tissue_positions.csv" else None,
        index_col=0,
    )
    coords.columns = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    # https://github.com/scverse/squidpy/issues/657
    coords.set_index(coords.index.astype(adata.obs.index.dtype), inplace=True)

    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    adata.obsm[Key.obsm.spatial] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    adata.obs.drop(columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True)

    if source_image_path is not None:
        source_image_path = Path(source_image_path).absolute()
        if not source_image_path.exists():
            logg.warning(f"Path to the high-resolution tissue image `{source_image_path}` does not exist")
        adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(source_image_path)

    return adata
    
    
    
    
    
def depth_first_cluster(adata, nn_radius, spatial_name, output_obs_name):
    df = pd.DataFrame(adata.obsm[spatial_name])
    cluster_out = adata.obs
    cluster_out["graph_cluster"] = -1

    knn_graph = neighbors.radius_neighbors_graph(df, radius=nn_radius)

    counter = -1
    queue = [random.randint(1,len(adata.obs.index))]
    visited = []


    options = adata.obs.index

    while len(options) > 0:

        current_node = options[ random.randint(0,len(options)-1) ]
        current_index = np.where(adata.obs.index == current_node)[0].tolist()
    
        queue = [ current_index[0] ]
    
        cn = 0
    
        counter += 1
    
        while len(queue) > 0:
        
            current_index = queue[0]
        
            current_connections = knn_graph[current_index,:].toarray()
        
            connections = np.where(current_connections == 1)[1].tolist()
        
            for i in connections: 
                if i not in visited + queue: 
                    queue.append(i)
      
            queue.remove(current_index)
        
            visited.append(current_index)
        
        
        
            cluster_out["graph_cluster"].iloc[current_index] = str(counter)
        
            cn+=1
    
        #Remove from options the ones that are visited
        options = options.drop(adata.obs.index[visited])
    
        visited = []
        
    adata.obs[output_obs_name] = cluster_out["graph_cluster"]

    return adata



def seperate_sections(adata, obs_column , spatial_name = 'spatial', nn_radius = 33):
    
    frame_dict = dict()
    
    adata.obs['sections'] = "nothing"
    
    #Remove unlabelled in obs_column
    adata = adata[adata.obs['basement_layer'] != "unlabelled"].copy()
    
    for i in adata.obs['basement_layer'].unique():
        
        current_sb = adata[adata.obs['basement_layer'] == i].copy()
        
        #depth first search to seperate the sections
        current_sb = depth_first_cluster(current_sb, nn_radius,spatial_name, "graph_cluster").copy()
        
        cluster_prefix = i
        
        cluster_name = cluster_prefix+"_section"

        cluster_labels = [cluster_prefix + x for x in current_sb.obs["graph_cluster"]]

        current_sb.obs[cluster_name] = cluster_labels
        
        #Remove the unlabelled IDs in basement_layer
        #current_sb = current_sb[current_sb.obs[obs_column] != "unlabelled"]
            
        #if the object has no spots left (because they were all unlabelled), do not save them in the dict
        
        pd_frame = current_sb.obs[cluster_name]

        frame_dict[i] = pd_frame
            
    
    adata.obs['sections'][ frame_dict[ adata.obs['basement_layer'].unique()[0] ].index ] = frame_dict[adata.obs['basement_layer'].unique()[0] ]

    adata.obs['sections'][ frame_dict[ adata.obs['basement_layer'].unique()[1] ].index ] = frame_dict[ adata.obs['basement_layer'].unique()[1] ]
    
    #adata = adata[adata.obs[obs_column] != "unlabelled"]
    
    return adata
    
    
    
    
#Make sure that your centre is a labelled and not unlabelled spot, otherwise it will get chucked out before this step and thus this function won't work
def match_and_merge(adata, basement_inside, centre_coordinates ,spatial_name = 'spatial', nn_radius = 33):
    adata.obs['basement_layer'].unique()
    
    upper_sections = [x for x in adata.obs["sections"].cat.categories.values.tolist() if "upper" in x]
    
    output = pd.DataFrame({"upper":["ClusterX"], "basement":["ClusterY"]})

    counter = 1
    
    for section in upper_sections:
    
        current_subset = adata[adata.obs['sections'] == section].copy()
    
        inverse_subset = adata[adata.obs['sections'] != section].copy()
    
        df = pd.DataFrame(inverse_subset.obsm[spatial_name])
    
        neigh = neighbors.KNeighborsClassifier(n_neighbors=5)
    
        neigh.fit(df, inverse_subset.obs["sections"])
    
        neigh.predict(current_subset.obsm[spatial_name])
    
        #As it's a roll, the villi will be sandwiched between two sections, and thus if there is three possible choices of section, 
        #We take the two sections which have the most connections, as assessed by kNN        
        options, options_count = np.unique(neigh.predict(current_subset.obsm[spatial_name]), return_counts=True)
        
        sort_idx = np.argsort(-options_count)

        options = options[sort_idx][0:2]                
            
        #Perhaps first check the nn results. If there is a broad majority then pick that, if it's slim do it based on distance

        if len(options)==1:
            output = pd.concat([output, pd.DataFrame({"upper":section, "basement":options[0]}, index = [counter]) ])
    
        else:
            option1_df = pd.DataFrame(adata[adata.obs['sections'] == options[0]].obsm[spatial_name])
            option2_df = pd.DataFrame(adata[adata.obs['sections'] == options[1]].obsm[spatial_name])
            
            option_1_cardinal_dist = scipy.spatial.distance_matrix(option1_df, centre_coordinates)
            option_2_cardinal_dist = scipy.spatial.distance_matrix(option2_df, centre_coordinates)
            
            options_score = [np.mean(option_1_cardinal_dist), np.mean(option_2_cardinal_dist)]
                        
            if basement_inside:
                basement_choice = options[np.where(options_score == np.min(options_score))]

            
            else:
                basement_choice = options[np.where(options_score == np.max(options_score))]
                                      
            
            output = pd.concat([output, pd.DataFrame( {"upper":section, "basement":basement_choice}, index = [counter] ) ])

    
        counter +=1
    
    
    output = output.drop(0)
    
    #Checks if any of the matches have less than 10 spots in them. If they do, drop them as they will not work with slingshot when getting the new length axis
    for i in range(1, len(output['upper'])+1 ):
                
        if np.shape(adata[adata.obs['sections']==output['upper'][i]])[0] < 10 or np.shape(adata[adata.obs['sections']==output['basement'][i]])[0] < 10:
            output=output.drop(i)
    
    
    output = output.reset_index()
    output = output.drop(columns = "index")
    
    return output




def getStartPoints(adata):
    cluster_choices = adata.obs['leiden_cluster'].cat.categories

    possible_starts = np.sum(adata.uns['paga']['connectivities'].toarray() > 0, axis = 1)

    cluster_choice = cluster_choices[np.where(possible_starts == 1)]
            
    return(cluster_choice)


def getLength(adata, start_cluster, graph_layout = 'kk'):
    from pyslingshot import Slingshot

    adata.obs['leiden_cluster'].cat.as_ordered()

    slingshot = Slingshot(adata, celltype_key="leiden_cluster", obsm_key="X_draw_graph_"+graph_layout, 
                      start_node=start_cluster, debug_level='verbose')

    slingshot.fit(num_epochs=1)
                
    pseudotime = slingshot.unified_pseudotime
        
    adata.obs["length"] = pseudotime
    
    return adata

    
def getDepth(adata, old_spatial_name = 'spatial', new_spatial_name = 'adjusted_spatial'):
    basement_spots = adata[adata.obs['basement_layer'] == "basement"].copy()
        
    #This distance matrix is to find which basement spots are closet to each other spot
    dist_mtx_3d = scipy.spatial.distance_matrix(adata.obsm[new_spatial_name],
                            basement_spots.obsm[new_spatial_name])
     
    villi_depth = np.min(dist_mtx_3d,axis = 1)
    
    #This distance matrix is where the 'depth' value comes from
    dist_mtx_2d = scipy.spatial.distance_matrix(adata.obsm[old_spatial_name],
                            basement_spots.obsm[old_spatial_name])
    
    col_index = np.argmin(dist_mtx_3d, axis = 1).tolist()
    row_index = list(range(0, len(np.argmin(dist_mtx_3d, axis = 1))))

    depth = dist_mtx_2d[row_index, col_index]
    
    #Scale the depth values so they are shorter than the length
    scale_max = np.max(adata.obs['length']) / 15
    depth = 0 + (scale_max - 0) * (villi_depth - np.min(villi_depth) ) / ( np.max(villi_depth) - np.min(villi_depth) )
    
    adata.obs['depth'] = villi_depth
    
    adata.obsm[new_spatial_name] =  np.c_[adata.obs['length'],
                                                  adata.obs['depth']]
    
    return adata
    
    
    
def scaleNewSpatial(adata_list, old_spatial_name = 'spatial', new_spatial_name = 'adjusted_spatial'):
    
    upper_section = adata_list[0]
    basement_section = adata_list[1]
    
    df_upper = pd.DataFrame(upper_section.obsm[old_spatial_name])
    
    distance_mtx_upper = scipy.spatial.distance_matrix(df_upper, df_upper)
    
    df_basement = pd.DataFrame(basement_section.obsm[old_spatial_name])

    distance_mtx_basement = scipy.spatial.distance_matrix(df_basement, df_basement)
        
    scale_max = ( ( np.max([np.max(upper_section.obs['length']), np.max(basement_section.obs['length'])]) / 100) *2 ) * np.max(scipy.spatial.distance_matrix(upper_section.obsm[old_spatial_name],            basement_section.obsm[old_spatial_name])) 
    
    upper_section.obs['length'] = 0 + (scale_max - 0) * (upper_section.obs['length'] - np.min(upper_section.obs['length']) ) / ( np.max(upper_section.obs['length']) - np.min(upper_section.obs['length']) ) 
    basement_section.obs['length'] = 0 + (scale_max - 0) * (basement_section.obs['length'] - np.min(basement_section.obs['length']) ) / (np.max(basement_section.obs['length']) - np.min(basement_section.obs['length']) )

    
    upper_section.obsm[new_spatial_name] =  np.c_[upper_section.obs['length'],
                                                  upper_section.obsm[old_spatial_name]]
    
    basement_section.obsm[new_spatial_name] =  np.c_[basement_section.obs['length'],
                                                  basement_section.obsm[old_spatial_name]]
    
    #The scaling doesn't fix all the issues with making the basement and upper lengths comparable. Sometimes (due to curving), one of the lengths
    #can be out of sync with the other. To solve this, we utilise kNN regression to force the lengths values to be similar based on their spatial context
    
    neigh = neighbors.KNeighborsRegressor(n_neighbors=5)
    neigh.fit(upper_section.obsm[new_spatial_name], upper_section.obs['length'])
    
    new_basement_length = neigh.predict(basement_section.obsm[new_spatial_name])
    
    basement_section.obs['length'] = new_basement_length
    
    basement_section.obsm[new_spatial_name] =  np.c_[basement_section.obs['length'],
                                                  basement_section.obsm[old_spatial_name]]
    
    merge_dataset = ad.concat([basement_section, upper_section], uns_merge= 'first')

    return merge_dataset    
    
    

def getNewSpatial(adata, match_df,old_spatial_name = 'spatial', new_spatial_name = 'adjusted_spatial', graph_layout = 'kk', leiden_resolution = 0.3):
    
    sections = adata.obs['sections'].unique()
        
    dataset_output = []
    
    for i in range(0, len(match_df['upper']) ):

        section_pairs = []
        
        possible_starts_df = pd.DataFrame({"upper":["ClusterX"], "basement":["ClusterY"]})

        counter = 0
                
        for j in [match_df['upper'][i], match_df['basement'][i]]:
            
            current_section = adata[adata.obs['sections'] == j].copy()
            
            sc.pp.neighbors(current_section, use_rep='spatial', 
                    metric = 'euclidean')

            leiden_visium(current_section, leiden_resolution)
        
            #If there is only one cluster repeat the clustering at a higher resolution until there is more than one
            
            new_resolution = leiden_resolution
            while len(current_section.obs['leiden_cluster'].unique()) < 2:
                leiden_visium(current_section, new_resolution)
                new_resolution +=0.1 

            cluster_results = current_section.obs['leiden_cluster']
            
            df = pd.DataFrame(current_section.obsm[old_spatial_name])
            
            
            #PySlingshot requires that the clusters be integer ordered categorical data
            new_clusters = pd.Series(cluster_results.astype(int), dtype = 'category').cat.as_ordered()
            current_section.obs['leiden_cluster'] = new_clusters

            knn_graph = neighbors.radius_neighbors_graph(df, radius=33,
                            include_self= True)

            #Replace the connectivities section so it is binary, 
            #Either the spots are next to each other and are connected, or not
            current_section.obsp['connectivities'] = knn_graph
    
            knn_graph = neighbors.radius_neighbors_graph(df, radius=33,
                            include_self= True, mode = 'distance')

            current_section.obsp['distances'] = knn_graph
            
            # sq.pl.spatial_scatter(current_section, color = "leiden_cluster")

            sc.tl.paga(current_section, groups='leiden_cluster')

            sc.pl.paga(current_section, color='leiden_cluster', plot = False)

            sc.tl.draw_graph(current_section, init_pos='paga', layout = graph_layout)

            sc.tl.diffmap(current_section, n_comps = 3)
            
            section_pairs.append(current_section)
        
        upper_starts = getStartPoints(section_pairs[0])
                
        basement_starts = getStartPoints(section_pairs[1])
        
        start_dist_df = pd.DataFrame({upper_starts[0]:[0,0], upper_starts[1]:[0,0]}, index = basement_starts)
        
#         print("start points upper:")
#         print(upper_starts)
        
#         print("start points basement:")
#         print(basement_starts)
        
        for upper in upper_starts:
            
            upper_section = section_pairs[0].copy()

            upper_section = upper_section[upper_section.obs['leiden_cluster'] == upper ].copy()

            upper_df = pd.DataFrame(upper_section.obsm[old_spatial_name])
            
            for basement in basement_starts:
                
                basement_section = section_pairs[1].copy()
                
                basement_section=basement_section[basement_section.obs['leiden_cluster'] == basement ].copy()
                
                basement_df = pd.DataFrame(basement_section.obsm[old_spatial_name])
            
                distance_result = np.mean( scipy.spatial.distance_matrix(upper_df, basement_df) )
                                
                start_dist_df[upper][basement] = distance_result
             
        #We take the clusters which had the smallest euclidean distance and calculate pseudotime with slingshot 
        #From these starting clusters
        
        start_dist_mtx = np.array(start_dist_df)
        
        #The 0th index is the upper and the 1st index is the basement start cluster
        start_clusters = [start_dist_df.columns[ np.where( start_dist_mtx == np.min(start_dist_mtx) )[0] ][0],
                         start_dist_df.index[ np.where( start_dist_mtx == np.min(start_dist_mtx) )[1] ][0]]
                
        basement_section = section_pairs[1][section_pairs[1].obs['sections']==match_df['basement'][i]].copy()
        upper_section = section_pairs[0][section_pairs[0].obs['sections']==match_df['upper'][i]].copy()
                
#         print("start clusters")
#         print(start_clusters)
        
#         print("upper dataset clusters")
#         print(upper_section.obs['leiden_cluster'].unique())
#         print(upper_section)
        
#         print("basement dataset clusters")
#         print(basement_section.obs['leiden_cluster'].unique())
#         print(basement_section)
        
        #Get the new lengths
        upper_section = getLength(upper_section, start_clusters[0])
        basement_section = getLength(basement_section, start_clusters[1])
        
        section_pairs[0] = upper_section.copy()
        section_pairs[1] = basement_section.copy()
        
        merge_adata = scaleNewSpatial(section_pairs)
        
        merge_adata = getDepth(merge_adata)

        dataset_output.append(merge_adata)
        
    return dataset_output



#Need a function which connects up the disparate sections into one common length axis 
#Have it input the order that the sections should be in

#It then connects things up 



#Create plotting function for length and depth

#Put a semi-transparent area graph according to location behind when plotting expression

def adjustedSpatial_scatter(adata, obs_value, obsm_space = 'spatial'):
    
    points = adata.obsm[obsm_space]
    
    #if there are multiple, iterate through them
    
    if type(obs_value) is str:
        obs_value = [obs_value]
            
    else:
        raise ValueError(
            "Invalid type"
        )
        
    for i in obs_value:
            
            
        #Check if the color is a gene or an obs column
        
        #If we want to colour by an obs metadata column
        if i in adata.obs.keys():
            
            
            
            x = adata.obsm[obsm_space][:,0]
            y = adata.obsm[obsm_space][:,1]
            color = np.array( adata.obs[i] )

            le = preprocessing.LabelEncoder()
            le.fit(color)
            color = le.transform(color)

            _, ax = plt.subplots()
            scatter = ax.scatter(x, y, c=color)
            _.set_figheight(5)
            _.set_figwidth(15)
            ax.set(xlabel= "Intestine length", ylabel="Intestine Depth")
            _ = ax.legend(
                scatter.legend_elements()[0], np.unique(adata.obs[i]), loc="lower right", title="Classes"
            )
            
            
        #If we want to colour by a genes expression
        elif i in adata.var_names:
            
            x = adata.obsm[obsm_space][:,0]
            y = adata.obsm[obsm_space][:,1]
            color = adata.X[:,np.where(adata.var_names == i)[0]].toarray()


            f, ax = plt.subplots()
            points = ax.scatter(x, y, c=color, s=50)
            ax.set(xlabel= "Intestine length", ylabel="Intestine Depth")
            f.colorbar(points).ax.set_ylabel('Normalized gene expression', rotation=270, fontsize = 10, labelpad=15)
            plt.title(i)
            
            f.set_figheight(5)
            f.set_figwidth(15)

            
        else:
            raise ValueError(
            "The colouring variable is not present in your adata.obs or adata.var_names"
        )
    
        
    
