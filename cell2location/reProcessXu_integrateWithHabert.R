library(Seurat)
library(stringr)

#Making sense of the datasets

#nonTB means no T cells or B cells

data <- ReadMtx(mtx = "/datastore3/RossOlympia/xu_intestineAtlas_2019/ncbi_data/GSE124880_PP_LP_mm10_count_matrix.mtx.gz",
                cells = "/datastore3/RossOlympia/xu_intestineAtlas_2019/ncbi_data/GSE124880_PP_LP_mm10_count_barcode.tsv.gz" ,
                features = "/datastore3/RossOlympia/xu_intestineAtlas_2019/ncbi_data/GSE124880_PP_LP_mm10_count_gene.tsv.gz", feature.column = 1)

cluster_cells <- read.table("/datastore3/RossOlympia/xu_intestineAtlas_2019/SCP210/metadata/Food-Allergy-PP-LP_cluster_v1.txt", header = 2)
cluster_cells <- cluster_cells [-1,]
cluster_cells$NAME <- str_replace_all(cluster_cells$NAME, "Food-Allergy-PP_", "")
cluster_cells$NAME <- str_replace_all(cluster_cells$NAME, "Food-Allergy-LP_", "")
row.names(cluster_cells) <- cluster_cells$NAME

embeddings_cells_pp <- read.table("/datastore3/RossOlympia/xu_intestineAtlas_2019/SCP210/cluster/Food-Allergy-PP_coord_2d.txt", header = 1)
embeddings_cells_pp <- embeddings_cells_pp[-1,]
embeddings_cells_pp$NAME <- str_replace_all(embeddings_cells_pp$NAME, "Food-Allergy-PP_", "")
row.names(embeddings_cells_pp) <- embeddings_cells_pp$NAME
embeddings_cells_pp$NAME <- NULL

embeddings_cells_lp <- read.table("/datastore3/RossOlympia/xu_intestineAtlas_2019/SCP210/cluster/Food-Allergy-LP_coord_2d.txt", header = 1)
embeddings_cells_lp <- embeddings_cells_lp[-1,]
embeddings_cells_lp$NAME <- str_replace_all(embeddings_cells_lp$NAME, "Food-Allergy-LP_", "")
row.names(embeddings_cells_lp) <- embeddings_cells_lp$NAME
embeddings_cells_lp$NAME <- NULL

embeddings_cells <- rbind(embeddings_cells_pp, embeddings_cells_lp)
embeddings_cells$X <- as.numeric(embeddings_cells$X)
embeddings_cells$Y <- as.numeric(embeddings_cells$Y)

lp_cells <- row.names(embeddings_cells_lp)

cluster_cells$location <- rep("PP", length(cluster_cells$Cluster))
cluster_cells$location[which(row.names(cluster_cells) %in% lp_cells)] <- "LP"

embeddings_cells <- embeddings_cells[order( match( row.names(embeddings_cells), colnames(data) ) ),]
cluster_cells <- cluster_cells[order( match( row.names(cluster_cells), colnames(data) ) ),]

#Get samples from cell names
samples <- str_replace(row.names(embeddings_cells), "^(.*)_.*$", "\\1")

cluster_cells$sample <- samples

colnames(embeddings_cells) <- c("tSNEscVI_1", "tSNEscVI_2")

cluster_cells$NAME <- NULL

intestine_obj <- CreateSeuratObject(data, meta.data = cluster_cells)
intestine_obj[["percent.mt"]] <- PercentageFeatureSet(intestine_obj, pattern = "^mt-")
intestine_obj <- NormalizeData(intestine_obj)

intestine_obj[['tsne.scVI']] <- CreateDimReducObject(embeddings = as.matrix(embeddings_cells), key = 'tSNEscVI_', assay = 'RNA')

DimPlot(intestine_obj, reduction = "tsne.scVI", group.by = "Cluster", label = T)

Idents(intestine_obj) <- "Cluster"

intestine_obj <- RenameIdents(intestine_obj,c("1" = "Resting Cd4+ T cell", "2" = "Resting B cell", "3" = "ILC3", "4" = "LTi cell",
                                 "5" = "Cd8+ T cell", "6" = "pDC", "7" = "Activated Cd4+ T cell", "8" = "NK cell", "9" = "GC B cell (DZ)",
                                 "10" = "GC B cell (LZ)", "11" = "ILC1", "12" = "ILC2", "13" = "Cd4+ T cell (low UMI)", "14" = "DC (Cd103+ Cd11b+",
                                 "15" = "ILC3 (low UMI)", "16" = "LTi cell (low UMI)", "17" = "gammaDelta T cell (Xcl1+)", "18" = "Unresolved_1", "19" = "NKT cell",
                                 "20" = "Plasma cell", "21" = "DC (Cd103+ Cd11b-)", "22" = "Macrophage", "23" = "Resting B cell (low UMI)", "24" = "DC (Cd103- C1)",
                                 "25" = "Endothelial cell", "26" = "Mast cell", "27" = "NK cell (low UMI)", "28" = "GC B cell (low UMI)", "29" = "Doublets_1",
                                 "30" = "Epithelial_cell_1", "31" = "DC (Cd103- C2)", "32" = "Stromal_cell (DN)", "33" = "Fibroblast", "34" = "Unresolved_2",
                                 "35" = "Epithelial_cell_2", "36" = "Neutrophil", "37" = "Unresolved_3", "38" = "gammaDelta T cell (Gzma+)", "39" = "Epithelial_cell (low UMI)",
                                 "40" = "Macrophage (low UMI)", "41" = "Doublets_2", "42" = "Doublets_3", "43" = "Doublets_4", "44" = "Basophil",
                                 "45" = "T precursor-like cell", "46" = "Lympahtic endothelial-like cell"))

intestine_obj$paperCellType <- Idents(intestine_obj)

DimPlot(intestine_obj, reduction = "tsne.scVI", label = T)

VlnPlot(intestine_obj, "nFeature_RNA", split.by = "location")
VlnPlot(intestine_obj, "nFeature_RNA", group.by = "sample")
VlnPlot(intestine_obj, "nCount_RNA", group.by = "sample")
VlnPlot(intestine_obj, "percent.mt", group.by = "sample")

intestine_obj$everything <- "project"

#Re QC data
#lower nFeature, upper nFeature, percent.mt
qc_list <- list("04242017_PP_location_WT_Allergy_v2_AI" = c(500, 1700, 10),
                "04242017_PP_location_WT_Allergy_v2_AJ" = c(200, 1500, 7),
                "04242017_PP_location_WT_Allergy_v2_CD" = c(200, 2900, 8),
                "04242017_PP_location_WT_Allergy_v2_CI" = c(250, 2200, 8),
                "04242017_PP_location_WT_Allergy_v2_CJ" = c(300, 3000, 8),
                "05152017_PP_location_WTAllergy_v2_AD" = c(200, 3000, 7),
                "05152017_PP_location_WTAllergy_v2_AI" = c(300, 2200, 7),
                "05152017_PP_location_WTAllergy_v2_AJ" = c(400, 2500, 6.5),
                "05152017_PP_location_WTAllergy_v2_CD" = c(300, 2700, 7),
                "05152017_PP_location_WTAllergy_v2_CI" = c(300, 2000, 6),
                "05152017_PP_location_WTAllergy_v2_CJ" = c(400, 2700,7),
                "06132017_PP_WT_Allergy_v2_allergy" = c(500, 3000, 7.5),
                "06132017_PP_WT_Allergy_v2_ctrl" = c(500, 2500, 6.5), 
                "07142017_PP_WT_Allergy_v2_Allergy" = c(300, 3000, 7.5),
                "07142017_PP_WT_Allergy_v2_Control" = c(300, 3000, 7.5),
                "09012017_PP_WT_Allergy_v2_Allergy_1" = c(300, 3000, 7.5),
                "09012017_PP_WT_Allergy_v2_Allergy_2" = c(400, 3000, 7.5),
                "09012017_PP_WT_Allergy_v2_Control_1" = c(200, 3000, 7.5),
                "09012017_PP_WT_Allergy_v2_Control_2" = c(300, 3100, 7),
                "10172017_ctrl_allergy_nonPP_nonTB_v2_Allergy_1" = c(250, 2200, 10),
                "10172017_ctrl_allergy_nonPP_nonTB_v2_Allergy_2" = c(250, 2400, 9),
                "10172017_ctrl_allergy_nonPP_nonTB_v2_Ctrl_1" = c(300, 2300, 9),
                "10172017_ctrl_allergy_nonPP_nonTB_v2_Ctrl_2" = c(250, 2200, 8), 
                "11292017_IgDnegSI_nonTBSI_v2_Allergy_IgDLow" = c(500, 2700, 8),
                "11292017_IgDnegSI_nonTBSI_v2_Allergy_nonTB"= c(300,2500, 8),
                "11292017_IgDnegSI_nonTBSI_v2_Ctrl_IgDLow" = c(300, 3000, 9),
                "11292017_IgDnegSI_nonTBSI_v2_Ctrl_nonTB" = c(300, 3000, 8))

intestine_split <- SplitObject(intestine_obj, "sample")

counter <- 1
for(i in names(intestine_split)){
  
  current <- intestine_split[[i]]
  print(VlnPlot(current, features = c("nFeature_RNA", "percent.mt"), group.by = "everything"))
  
  current <- subset(current, nFeature_RNA > qc_list[[i]][1] &
                      nFeature_RNA < qc_list[[i]][2] &
                      percent.mt < qc_list[[i]][3])
  print(i)
  if(counter == 28){
    print(i)
    print(VlnPlot(current, features = c("nFeature_RNA", "percent.mt"), group.by = "everything"))
    stop()
  }
  
  counter <- counter + 1
  
}

intestine_obj <- merge(intestine_split[[1]], intestine_split[[2]])
metaData <- intestine_split[[1]]@meta.data
reduced_dim <- intestine_split[[1]]@reductions$tsne.scVI@cell.embeddings

metaData <- rbind(metaData, intestine_split[[2]]@meta.data)
reduced_dim <- rbind(reduced_dim, intestine_split[[2]]@reductions$tsne.scVI@cell.embeddings)

for(i in 3:length(intestine_split)){
  
  intestine_obj <- merge(intestine_obj, intestine_split[[i]])
  metaData <- rbind(metaData, intestine_split[[i]]@meta.data)
  reduced_dim <- rbind(reduced_dim, intestine_split[[i]]@reductions$tsne.scVI@cell.embeddings)

  
}

intestine_obj@meta.data <- metaData
intestine_obj[['tsne.scVI']] <- CreateDimReducObject(embeddings = as.matrix(reduced_dim), key = 'tSNEscVI_', assay = 'RNA')


DimPlot(intestine_obj, reduction = "tsne.scVI", label = T)

#Keep only immune cells and remove doublets
#Also removed the unresolved and the low UMI GC B cells (the latter was because it didn't say whether they were light or dark zone)

intestine_immune_obj <- subset(intestine_obj, paperCellType %in% c("Resting Cd4+ T cell", "Resting B cell","ILC3","LTi cell",
                                                            "Cd8+ T cell", "pDC", "Activated Cd4+ T cell", "NK cell",  "GC B cell (DZ)",
                                                             "GC B cell (LZ)",  "ILC1", "ILC2", "Cd4+ T cell (low UMI)", "DC (Cd103+ Cd11b+",
                                                            "ILC3 (low UMI)",  "LTi cell (low UMI)",  "gammaDelta T cell (Xcl1+)", "NKT cell",
                                                            "Plasma cell",  "DC (Cd103+ Cd11b-)", "Macrophage",  "Resting B cell (low UMI)", "DC (Cd103- C1)",
                                                             "Mast cell",  "NK cell (low UMI)",  
                                                              "DC (Cd103- C2)",
                                                             "Neutrophil",  "gammaDelta T cell (Gzma+)", 
                                                           "Macrophage (low UMI)", "Basophil",
                                                             "T precursor-like cell"))


#Merge the low UMI and normal clusters together

intestine_immune_obj <-  RenameIdents(object = intestine_immune_obj,
                                                      "Cd4+ T cell (low UMI)" = "Resting Cd4+ T cell",
                                                      "LTi cell (low UMI)" = "LTi cell",
                                                      "NK cell (low UMI)" = "NK cell",
                                                      "Resting B cell (low UMI)" = "Resting B cell",
                                                      "ILC3 (low UMI)" = "ILC3",
                                                      "Macrophage (low UMI)" = "Macrophage")

DimPlot(intestine_immune_obj, reduction = "tsne.scVI", label = T)

#CD103, Cd11b
VlnPlot(intestine_immune_obj, features = c("Itgae", "Itgam"))

VlnPlot(intestine_immune_obj, features = c("Izumo1r"))


# intestine_immune_markers <- FindAllMarkers(intestine_immune_obj, logfc.threshold = 0.5, only.pos = T)
# 
# intestine_immune_markers %>%
#   group_by(cluster) %>%
#   dplyr::filter(avg_log2FC > 1) %>%
#   slice_head(n = 10) %>%
#   ungroup() -> top10

intestine_immune_obj$paperCellType <- Idents(intestine_immune_obj)

intestine_immune_obj$paperCellType <-as.character(intestine_immune_obj$paperCellType)

DimPlot(intestine_immune_obj, reduction = "tsne.scVI", label = T, group.by = "paperCellType")
DimPlot(intestine_immune_obj, reduction = "tsne.scVI", label = T, group.by = "paperCellType", split.by = "sample")

#Put in categorical effects into meta data

#Donor ID

#This commands get the first value in each list of a list of lists
donor_id <- sapply(str_split( intestine_immune_obj$sample, "_" ), "[[", 1)

#All or no T and B cells or no B cells
TB_id <- rep("immune_all", dim(intestine_immune_obj)[2])
TB_id[str_detect(intestine_immune_obj$sample, "nonTB")] <- "noTB"
TB_id[str_detect(intestine_immune_obj$sample, "IgD")] <- "noB"

#Control vs allergy
ctrl_allergy_id <- rep("allergy", dim(intestine_immune_obj)[2])
ctrl_allergy_id[str_detect(intestine_immune_obj$sample, "Ctrl|Control")] <- "Control"

intestine_immune_obj$donor_id <- donor_id
intestine_immune_obj$type <- TB_id
intestine_immune_obj$condition <- ctrl_allergy_id


unique(intestine_immune_obj$sample)

intestine_immune_obj <- RenameIdents(intestine_immune_obj,c("Resting Cd4+ T cell" = "Cd4+ T cell",  "Resting B cell" = "B cell",  "ILC3", "LTi cell",
                                              "GC B cell (DZ)" = "GC B cell",
                                               "GC B cell (LZ)" = "GC B cell", "DC (Cd103+ Cd11b+" = "DC",
                                                 "gammaDelta T cell (Xcl1+)" = "gammaDelta T cell", 
                                               "Plasma cell",  "DC (Cd103+ Cd11b-)" = "DC",  "DC (Cd103- C1)" = "DC",
                                               "DC (Cd103- C2)" = "DC", 
                                                "Neutrophil",   "gammaDelta T cell (Gzma+)" = "gammaDelta T cell",  "Basophil",
                                               "T precursor-like cell"))

intestine_immune_obj$simpleCellType <- Idents(intestine_immune_obj)

x <- FindAllMarkers(intestine_immune_obj, only.pos = T)
pdc <- subset(x, cluster == "pDC")
mast <- subset(x, cluster == "Mast cell")


VlnPlot(intestine_immune_obj, features = c("Siglech", "Klk1", "Cox6a2"))


#Removed the T cell precursor cells as there wasn't many of them, yet they showed up massively in the cell2location output. 
#I think this is because they don't have a particularly good transcriptome
intestine_immune_obj <- subset(intestine_immune_obj, simpleCellType != "T precursor-like cell")

DimPlot(intestine_immune_obj, group.by = "simpleCellType", label = T)

saveRDS(intestine_immune_obj, "/datastore3/RossOlympia/xu_intestineAtlas_2019/dataset_subset.rds")

#Convert object to annData
library(SeuratDisk)

SaveH5Seurat(intestine_immune_obj, filename = "/datastore3/RossOlympia/xu_intestineAtlas_2019/intestine_immune.h5Seurat",
             overwrite = T)
Convert("/datastore3/RossOlympia/xu_intestineAtlas_2019/intestine_immune.h5Seurat", dest = "h5ad",
        overwrite = T)

#Merge with Habert

habert_counts <- read.csv("/datastore3/RossOlympia/Habert_Epithelia/habert_epithelial_raw_counts.csv", header = 1, row.names = 1)

habert_meta <- read.csv("/datastore3/RossOlympia/Habert_Epithelia/habert_epithelial_metadata.csv", header = 1, row.names = 1)

habert_tsne <- read.table("/datastore3/RossOlympia/Habert_Epithelia/Fig6_tSNE.txt", row.names = 1)

habert_tsne <- habert_tsne[3:dim(habert_tsne[1]),]

habert_tsne <- habert_tsne[row.names(habert_meta),]

colnames(habert_tsne) <- c("tSNE_1", "tSNE_2")

habert_tsne$tSNE_1 <- as.numeric(habert_tsne$tSNE_1)
habert_tsne$tSNE_2 <- as.numeric(habert_tsne$tSNE_2)

habert_obj <- CreateSeuratObject(habert_counts, meta.data = habert_meta)

habert_obj[['tsne']] <- CreateDimReducObject(embeddings = as.matrix(habert_tsne), key = 'tSNE_', assay = 'RNA')

DimPlot(habert_obj, reduction = "tsne")
DimPlot(habert_obj, reduction = "tsne", group.by = "Celltype")

habert_obj$type <- "non_immune" 
habert_obj$simpleCellType <- habert_obj$Celltype
habert_obj$paperCellType <- habert_obj$Celltype
habert_obj$sample <- habert_obj$orig.ident
habert_obj$condition <- habert_obj$orig.ident


#type: shows what cell types should be present: immune_all, noB, noTB and non_immune
#sample: sequencing run 
#simpleCellType: cell types with simple immune cell types
#paperCellType: cell types with complex immune cell types
#condition: what condition is present in the sample

merge_obj <- merge(intestine_immune_obj, y = c(habert_obj))

saveRDS(merge_obj, "/datastore3/RossOlympia/xu_habert_merge_obj.rds")
SaveH5Seurat(merge_obj, filename = "/datastore3/RossOlympia/xu_habert_merge_obj.h5Seurat",
             overwrite = T)
Convert("/datastore3/RossOlympia/xu_habert_merge_obj.h5Seurat", dest = "h5ad",
        overwrite = T)
