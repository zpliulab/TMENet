#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) stop("Usage: run_custom_batch.R protocol.json OUTPUT_ROOT CASE_LIST WORKERS")
code_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])))
source(file.path(code_dir, "custom_core.R"))
suppressPackageStartupMessages(library(jsonlite))
cfg <- fromJSON(args[1])
.libPaths(c(cfg$private_lib, .libPaths()))
suppressPackageStartupMessages({library(nichenetr); library(Seurat); library(dplyr)})
options(dplyr.summarise.inform = FALSE)
output_root <- args[2]; keys <- readLines(args[3]); workers <- as.integer(args[4])
protocol_md5 <- unname(tools::md5sum(args[1]))
stopifnot(workers >= 1L, all(grepl("^geneNum_[0-9]+_id_[0-9]+/cellType[234]/seed(12|22|32|42|52)$", keys)))
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_root, "sessionInfo.txt"))

write_manifest <- function(x, dest) {
  temporary <- paste0(dest, ".tmp.", Sys.getpid())
  write_json(x, temporary, pretty = TRUE, auto_unbox = TRUE, null = "null", na = "null")
  if (!file.rename(temporary, dest)) stop("Atomic manifest rename failed")
}

run_case_inner <- function(key, dest) {
  started <- Sys.time()
  input_file <- function(name) {
    if (!name %in% cfg$allowed_inference_files) stop("Forbidden inference input: ", name)
    file.path(cfg$sim_root, key, name)
  }
  counts <- read_numeric_matrix(input_file("rnaseq_data.txt"))
  original_genes <- rownames(counts)
  alias <- setNames(paste0("G", seq_along(original_genes)), original_genes)
  inverse <- setNames(original_genes, alias)
  meta <- read.delim(input_file("metadata_fine.tsv"), colClasses = "character", check.names = FALSE)
  if (!all(c("cell", "cell_type") %in% names(meta)) || anyDuplicated(meta$cell) || !setequal(meta$cell, colnames(counts))) stop("Metadata alignment failed")
  meta <- meta[match(colnames(counts), meta$cell), , drop = FALSE]
  types <- unique(meta$cell_type)
  type_alias <- setNames(paste0("CT", seq_along(types)), types)
  type_inverse <- setNames(types, type_alias)
  meta$nn_type <- unname(type_alias[meta$cell_type]); rownames(meta) <- meta$cell
  lr <- read.delim(input_file("LR_candidates_input.tsv"), colClasses = "character", check.names = FALSE)
  if (!setequal(names(lr), c("ligand", "receptor"))) stop("LR input must contain identities only; effect/truth forbidden")
  lr <- unique(lr[, c("ligand", "receptor"), drop = FALSE])
  if (!all(c(lr$ligand, lr$receptor) %in% original_genes)) stop("LR genes absent from RNA")
  write_tsv(data.frame(original = original_genes, internal = unname(alias)), file.path(dest, "gene_mapping.tsv"))
  atac <- read_numeric_matrix(input_file("atac_data.txt"))
  rg <- read_numeric_matrix(input_file("atac_region2gene.txt")); rt <- read_numeric_matrix(input_file("atac_region2tf.txt"))
  if (!setequal(colnames(atac), colnames(counts))) stop("ATAC/RNA cell labels differ")
  gr <- reconstruct_atac_gr(atac, rg, rt, original_genes)
  library_size <- colSums(counts); library_size[library_size == 0] <- 1
  normalized <- log1p(t(t(counts)/library_size)*10000)
  rho <- partial_rank_correlation(normalized, meta$cell_type)
  active_tfs <- rownames(gr)[rowSums(gr) > 0]
  bridge <- build_proxy_edges(rho, unique(lr$receptor), active_tfs)
  positions <- which(gr > 0, arr.ind = TRUE)
  gr_edges <- data.frame(from = rownames(gr)[positions[, 1]], to = colnames(gr)[positions[, 2]], weight = gr[positions])
  write_tsv(gr_edges, file.path(dest, "atac_tf_target_prior.tsv"))
  write_tsv(bridge, file.path(dest, "receptor_tf_association_proxy.tsv"))
  input_hash <- as.list(tools::md5sum(vapply(cfg$allowed_inference_files, input_file, character(1))))
  base <- list(case = key, method = cfg$method, version = as.character(packageVersion("nichenetr")), upstream_commit = cfg$upstream_commit, protocol_md5 = protocol_md5, input_md5 = input_hash, no_truth_inputs = TRUE, input_genes = nrow(counts), input_cells = ncol(counts), fine_cell_types = length(types), lr_candidates = nrow(lr), atac_prior_edges = nrow(gr_edges), receptor_tf_proxy_edges = nrow(bridge), bridge_top_k = max(1L, as.integer(ceiling(sqrt(length(active_tfs))))), bridge_is_curated_or_causal = FALSE, primary_lr_score = "official_generate_prioritization_tables_one_condition_custom_prior")
  finish_empty <- function(reason) {
    write_tsv(empty_lr_score(), file.path(dest, "lr_score.tsv"))
    write_manifest(c(base, list(status = "complete_no_prediction", reason = reason, elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")))), file.path(dest, "manifest.json"))
    list(case = key, status = "complete_no_prediction")
  }
  if (!nrow(gr_edges) || !nrow(lr)) return(finish_empty("no_ATAC_regulatory_edges_or_LR_candidates"))
  lr_net <- data.frame(from = unname(alias[lr$ligand]), to = unname(alias[lr$receptor]), source = "unlabeled_simulation_candidates")
  lr_sig <- rbind(data.frame(from = lr$ligand, to = lr$receptor, weight = 1), bridge[, c("from", "to", "weight"), drop = FALSE])
  lr_sig <- aggregate(weight ~ from + to, lr_sig, sum)
  lr_sig$from <- unname(alias[lr_sig$from]); lr_sig$to <- unname(alias[lr_sig$to])
  gr_edges_internal <- gr_edges
  gr_edges_internal$from <- unname(alias[gr_edges$from]); gr_edges_internal$to <- unname(alias[gr_edges$to])
  weighted <- apply_hub_corrections(list(lr_sig = lr_sig, gr = gr_edges_internal), lr_sig_hub = cfg$hub_correction_lr_sig, gr_hub = cfg$hub_correction_gr)
  raw_ltm <- construct_ligand_target_matrix(weighted, lr_net, ligands = as.list(sort(unique(lr_net$from))), algorithm = cfg$algorithm, damping_factor = cfg$damping_factor, ltf_cutoff = cfg$ltf_cutoff, secondary_targets = cfg$secondary_targets, remove_direct_links = cfg$remove_direct_links)
  ltm <- matrix(0, nrow(counts), ncol(raw_ltm), dimnames = list(unname(alias), colnames(raw_ltm)))
  ltm[rownames(raw_ltm), colnames(raw_ltm)] <- raw_ltm
  if (any(!is.finite(ltm)) || any(ltm < 0)) stop("Invalid native ligand-target prior")
  saveRDS(list(ligand_target_matrix = ltm, weighted_networks = weighted, lr_network = lr_net), file.path(dest, "custom_prior.rds"))
  base$nonzero_ligand_target_entries <- sum(ltm > 0)
  if (!any(ltm > 0)) return(finish_empty("native_propagation_produced_no_ligand_target_support"))
  rownames(counts) <- unname(alias)
  object <- CreateSeuratObject(counts = counts, meta.data = meta, min.cells = 0, min.features = 0)
  object <- NormalizeData(object, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
  Idents(object) <- object$nn_type
  markers <- calculate_de(object, "nn_type", min.pct = 0, logfc.threshold = 0, return.thresh = Inf, only.pos = FALSE, test.use = "wilcox")
  write_tsv(markers, file.path(dest, "native_celltype_markers.tsv"))
  expressed <- setNames(lapply(unname(type_alias), function(ct) get_expressed_genes(ct, object, pct = cfg$expression_pct)), unname(type_alias))
  activities <- list(); response_rows <- list(); status_rows <- list()
  for (receiver in unname(type_alias)) {
    background <- intersect(expressed[[receiver]], rownames(ltm))
    response <- select_response_genes(markers, receiver, background, cfg$response_fraction)
    ligands <- intersect(unique(lr_net$from[lr_net$to %in% expressed[[receiver]]]), unique(unlist(expressed)))
    if (length(background) < 2 || !length(response) || !length(setdiff(background, response)) || !length(ligands)) {
      status_rows[[receiver]] <- data.frame(receiver = receiver, status = "no_response_or_eligible_ligand", background_genes = length(background), response_genes = length(response), ligands = length(ligands))
      next
    }
    activity <- predict_ligand_activities(geneset = response, background_expressed_genes = background, ligand_target_matrix = ltm, potential_ligands = ligands)
    if (any(!is.finite(as.matrix(activity[, c("auroc", "aupr", "aupr_corrected")])))) stop("Non-finite official activity ranking metrics")
    constant <- vapply(activity$test_ligand, function(l) length(unique(ltm[background, l])) < 2, logical(1))
    if (any(!is.finite(activity$pearson) & !constant)) stop("Non-finite nonconstant ligand correlation")
    activity$constant_ligand_prior <- constant
    # Pearson NA for a constant predictor is retained in native output; LR
    # priority uses finite native aupr_corrected, not this auxiliary field.
    activity$receiver <- receiver
    activities[[receiver]] <- activity
    response_rows[[receiver]] <- data.frame(receiver = receiver, gene = background, response = background %in% response)
    status_rows[[receiver]] <- data.frame(receiver = receiver, status = "scored", background_genes = length(background), response_genes = length(response), ligands = length(ligands))
  }
  write_tsv(bind_rows(status_rows), file.path(dest, "receiver_status.tsv"))
  if (!length(activities)) return(finish_empty("no_receiver_has_non_degenerate_response_and_ligand_set"))
  activity_table <- bind_rows(activities)
  info <- process_table_to_ic(get_exprs_avg(object, "nn_type", assay_oi = "RNA"), table_type = "expression", lr_network = lr_net)
  de <- process_table_to_ic(markers, table_type = "celltype_DE", lr_network = lr_net, senders_oi = unname(type_alias), receivers_oi = names(activities))
  priority <- generate_prioritization_tables(info, de, activity_table, scenario = "one_condition")
  if (nrow(priority)) {
    keep <- mapply(function(s, r, l, rec) l %in% expressed[[s]] && rec %in% expressed[[r]], priority$sender, priority$receiver, priority$ligand, priority$receptor)
    priority <- priority[keep, , drop = FALSE]
  }
  if (any(!is.finite(priority$prioritization_score))) stop("Non-finite native LR prioritization score")
  score <- data.frame(sender_type = unname(type_inverse[priority$sender]), receiver_type = unname(type_inverse[priority$receiver]), ligand = unname(inverse[priority$ligand]), receptor = unname(inverse[priority$receptor]), score = priority$prioritization_score)
  if (anyNA(score) || anyDuplicated(score[, c("sender_type", "receiver_type", "ligand", "receptor")])) stop("LR output key validation failed")
  write_tsv(score, file.path(dest, "lr_score.tsv"))
  write_tsv(activity_table, file.path(dest, "native_ligand_activities.tsv"))
  write_tsv(priority, file.path(dest, "native_lr_prioritization.tsv"))
  write_tsv(bind_rows(response_rows), file.path(dest, "receiver_response_sets.tsv"))
  write_manifest(c(base, list(status = "complete", output_lr_rows = nrow(score), scored_receivers = length(activities), zero_prediction_receivers = length(types)-length(activities), constant_prior_activity_rows = sum(activity_table$constant_ligand_prior), elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")))), file.path(dest, "manifest.json"))
  list(case = key, status = "complete")
}

run_case <- function(key) {
  dest <- file.path(output_root, key)
  manifest <- file.path(dest, "manifest.json")
  if (file.exists(manifest)) {
    old <- fromJSON(manifest)
    if (!identical(old$protocol_md5, protocol_md5)) stop("Protocol mismatch on resume")
    if (old$status %in% c("complete", "complete_no_prediction")) return(list(case = key, status = old$status, resumed = TRUE))
  }
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  log <- file(file.path(dest, "run.log"), open = "at")
  sink(log); sink(log, type = "message")
  on.exit({sink(type = "message"); sink(); close(log); gc(verbose = FALSE)}, add = TRUE)
  tryCatch(run_case_inner(key, dest), error = function(err) {
    write_manifest(list(case = key, status = "failed", error = conditionMessage(err), call = paste(deparse(conditionCall(err)), collapse = " "), protocol_md5 = protocol_md5), file.path(dest, "failure.json"))
    list(case = key, status = "failed", error = conditionMessage(err))
  })
}
started <- Sys.time()
result <- parallel::mclapply(keys, run_case, mc.cores = workers, mc.preschedule = FALSE, mc.set.seed = TRUE)
write_manifest(list(cases = length(keys), statuses = as.list(table(vapply(result, function(x) if (is.list(x)) x$status else "worker_error", character(1)))), elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs"))), file.path(output_root, "batch_status.json"))
if (any(vapply(result, function(x) !is.list(x) || identical(x$status, "failed"), logical(1)))) quit(status = 1)
