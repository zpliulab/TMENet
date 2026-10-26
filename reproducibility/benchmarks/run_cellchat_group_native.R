#!/usr/bin/env Rscript

# Native-resolution CellChat runner for the canonical CCC simulations.
# It saves both directed cell-type-by-cell-type aggregate probabilities and
# native cell-type-pair-by-LR probabilities. Evaluation is performed without
# expanding group scores to artificial cell-pair scores.

suppressPackageStartupMessages({
  library(CellChat)
  library(jsonlite)
})

create_custom_db <- function(lr_pairs) {
  colnames(lr_pairs) <- tolower(trimws(colnames(lr_pairs)))
  colnames(lr_pairs)[colnames(lr_pairs) == "regulator"] <- "ligand"
  colnames(lr_pairs)[colnames(lr_pairs) == "target"] <- "receptor"
  if (!all(c("ligand", "receptor") %in% colnames(lr_pairs))) {
    stop("LR table lacks ligand/receptor columns")
  }
  pairs <- unique(data.frame(
    ligand = as.character(lr_pairs$ligand),
    receptor = as.character(lr_pairs$receptor),
    stringsAsFactors = FALSE
  ))
  pairs <- pairs[nzchar(pairs$ligand) & nzchar(pairs$receptor), , drop = FALSE]
  interaction_name <- make.unique(paste0(pairs$ligand, "_", pairs$receptor))
  interaction <- data.frame(
    interaction_name = interaction_name,
    pathway_name = paste0("simulated_pathway_", seq_len(nrow(pairs))),
    ligand = pairs$ligand,
    receptor = pairs$receptor,
    agonist = "",
    antagonist = "",
    co_A_receptor = "",
    co_I_receptor = "",
    annotation = "Secreted Signaling",
    evidence = "simulated_LR_identity",
    stringsAsFactors = FALSE,
    row.names = interaction_name
  )
  empty_complex <- data.frame(
    subunit_1 = character(), subunit_2 = character(),
    subunit_3 = character(), subunit_4 = character()
  )
  empty_cofactor <- data.frame(
    cofactor1 = character(), cofactor2 = character(),
    cofactor3 = character(), cofactor4 = character()
  )
  list(
    interaction = interaction,
    complex = empty_complex,
    cofactor = empty_cofactor,
    geneInfo = data.frame(Symbol = unique(c(pairs$ligand, pairs$receptor)), stringsAsFactors = FALSE)
  )
}

read_lr <- function(case_dir, lr_root, gene_num, dataset_id) {
  case_path <- file.path(case_dir, "LR_candidate_universe.tsv")
  tsv_path <- file.path(lr_root, sprintf("geneNum_%s_%s.txt", gene_num, dataset_id))
  csv_path <- file.path(lr_root, sprintf("lr_pairs_%s_%s.csv", gene_num, dataset_id))
  if (file.exists(case_path)) {
    read.table(case_path, sep = "\t", header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)
  } else if (file.exists(tsv_path)) {
    read.table(tsv_path, sep = "\t", header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)
  } else if (file.exists(csv_path)) {
    read.csv(csv_path, header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)
  } else {
    stop("simulated LR table is missing")
  }
}

write_empty_lr_score <- function(output_dir) {
  empty <- data.frame(
    sender_type = character(), receiver_type = character(),
    ligand = character(), receptor = character(), score = numeric(),
    stringsAsFactors = FALSE
  )
  write.table(
    empty, file.path(output_dir, "lr_score.tsv"), sep = "\t",
    quote = FALSE, row.names = FALSE
  )
}

write_lr_score <- function(object, probability, output_dir) {
  lr_sig <- object@LR$LRsig
  if (is.null(lr_sig) || nrow(lr_sig) == 0) {
    write_empty_lr_score(output_dir)
    return(invisible(NULL))
  }
  if (length(dim(probability)) != 3 || is.null(dimnames(probability)[[3]])) {
    stop("CellChat probability array lacks interaction identities")
  }
  interaction_ids <- as.character(dimnames(probability)[[3]])
  lr_ids <- rownames(lr_sig)
  if (is.null(lr_ids) || any(!interaction_ids %in% lr_ids)) {
    stop("CellChat probability interactions do not align with LRsig")
  }
  lr_lookup <- lr_sig[interaction_ids, , drop = FALSE]
  if (!all(c("ligand", "receptor") %in% colnames(lr_lookup))) {
    stop("CellChat LRsig lacks ligand/receptor identities")
  }
  long <- as.data.frame.table(probability, responseName = "score", stringsAsFactors = FALSE)
  colnames(long)[1:3] <- c("sender_type", "receiver_type", "interaction_id")
  long$ligand <- as.character(lr_lookup[as.character(long$interaction_id), "ligand"])
  long$receptor <- as.character(lr_lookup[as.character(long$interaction_id), "receptor"])
  long <- long[, c("sender_type", "receiver_type", "ligand", "receptor", "score")]
  write.table(
    long, file.path(output_dir, "lr_score.tsv"), sep = "\t",
    quote = FALSE, row.names = FALSE
  )
}

process_case <- function(case_key, sim_root, lr_root, output_root) {
  match <- regmatches(
    case_key,
    regexec("^geneNum_([0-9]+)_id_([0-9]+)/cellType([0-9]+)/seed([0-9]+)$", case_key)
  )[[1]]
  if (length(match) != 5) stop(sprintf("invalid case key: %s", case_key))
  gene_num <- match[2]
  dataset_id <- match[3]
  seed_value <- as.integer(match[5])
  case_dir <- file.path(sim_root, case_key)
  output_dir <- file.path(output_root, case_key)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  counts <- as.matrix(read.table(
    file.path(case_dir, "rnaseq_data.txt"), sep = "\t", header = TRUE,
    row.names = 1, check.names = FALSE
  ))
  storage.mode(counts) <- "double"
  fine_case_path <- file.path(case_dir, "metadata_fine.tsv")
  fine_metadata_root <- Sys.getenv("SCMULTISIM_FINE_METADATA_ROOT", "")
  metadata_resolution <- "legacy_coarse_label"
  if (file.exists(fine_case_path)) {
    fine <- read.table(fine_case_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
    if (!all(c("cell", "cell_type") %in% colnames(fine))) stop("fine metadata lacks cell/cell_type")
    metadata_raw <- data.frame(cell = as.character(fine$cell), labels = as.character(fine$cell_type))
    metadata_resolution <- "scMultiSim_exact_fine_state"
  } else if (nzchar(fine_metadata_root)) {
    fine_path <- file.path(fine_metadata_root, case_key, "metadata_fine.tsv")
    if (!file.exists(fine_path)) fine_path <- file.path(fine_metadata_root, case_key, "cell_metadata.tsv")
    fine <- read.table(fine_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
    if (!all(c("cell", "cell_type") %in% colnames(fine))) stop("fine metadata lacks cell/cell_type")
    metadata_raw <- data.frame(cell = as.character(fine$cell), labels = as.character(fine$cell_type))
    metadata_resolution <- "external_fine_state"
  } else {
    metadata_raw <- read.table(
      file.path(case_dir, "metadata.txt"), sep = "\t", header = FALSE,
      col.names = c("cell", "labels"), stringsAsFactors = FALSE
    )
  }
  metadata <- data.frame(
    labels = factor(metadata_raw$labels),
    row.names = as.character(metadata_raw$cell)
  )
  coordinates <- as.matrix(read.csv(
    file.path(case_dir, "coords.csv"), row.names = 1, check.names = FALSE
  ))
  storage.mode(coordinates) <- "double"
  common_cells <- Reduce(intersect, list(colnames(counts), rownames(metadata), rownames(coordinates)))
  if (length(common_cells) != ncol(counts)) stop("counts, metadata, and coordinates are not label-aligned")
  counts <- counts[, common_cells, drop = FALSE]
  metadata <- metadata[common_cells, , drop = FALSE]
  coordinates <- coordinates[common_cells, seq_len(2), drop = FALSE]

  library_sizes <- colSums(counts)
  library_sizes[library_sizes == 0] <- 1
  normalized <- log1p(t(t(counts) / library_sizes) * 10000)
  lr_pairs <- read_lr(case_dir, lr_root, gene_num, dataset_id)
  lr_names <- tolower(trimws(colnames(lr_pairs)))
  lr_names[lr_names == "regulator"] <- "ligand"
  lr_names[lr_names == "target"] <- "receptor"
  colnames(lr_pairs) <- lr_names
  original_lr_pair_count <- nrow(unique(lr_pairs[, c("ligand", "receptor"), drop = FALSE]))
  expressed_genes <- rownames(normalized)[rowSums(normalized) > 0]
  lr_pairs <- unique(lr_pairs[
    as.character(lr_pairs$ligand) %in% expressed_genes &
      as.character(lr_pairs$receptor) %in% expressed_genes,
    ,
    drop = FALSE
  ])

  if (nrow(lr_pairs) == 0) {
    levels <- levels(metadata$labels)
    group_score <- matrix(0, nrow = length(levels), ncol = length(levels), dimnames = list(levels, levels))
    write.table(
      group_score,
      file.path(output_dir, "group_score.tsv"),
      sep = "\t", quote = FALSE, col.names = NA
    )
    write_empty_lr_score(output_dir)
    write_json(
      list(
        method = "CellChat_2.2.0",
        case = case_key,
        resolution = "directed_cell_type_pair",
        distance_use = TRUE,
        original_lr_pair_count = original_lr_pair_count,
        valid_lr_pair_count = 0,
        zero_score_reason = "no LR pair has nonzero expression for both genes",
        cell_count = ncol(counts),
        cell_type_resolution = metadata_resolution,
        lr_candidate_source = if (file.exists(file.path(case_dir, "LR_candidate_universe.tsv"))) "case_LR_candidate_universe" else "legacy_global_LR"
      ),
      file.path(output_dir, "manifest.json"),
      pretty = TRUE, auto_unbox = TRUE
    )
    return(invisible(NULL))
  }

  set.seed(seed_value)
  spatial_factors <- data.frame(ratio = 1, tol = 0.5)
  object <- createCellChat(
    object = normalized,
    meta = metadata,
    group.by = "labels",
    datatype = "spatial",
    coordinates = coordinates,
    spatial.factors = spatial_factors
  )
  object@DB <- create_custom_db(lr_pairs)
  object <- subsetData(object)
  if (nrow(object@data.signaling) == 0) stop("no simulated LR genes retained by CellChat")
  object <- identifyOverExpressedGenes(object)
  object <- identifyOverExpressedInteractions(object)
  if (is.null(object@LR$LRsig) || nrow(object@LR$LRsig) == 0) {
    levels <- levels(metadata$labels)
    group_score <- matrix(0, nrow = length(levels), ncol = length(levels), dimnames = list(levels, levels))
    write.table(
      group_score,
      file.path(output_dir, "group_score.tsv"),
      sep = "\t", quote = FALSE, col.names = NA
    )
    write_empty_lr_score(output_dir)
    write_json(
      list(
        method = "CellChat_2.2.0",
        case = case_key,
        resolution = "directed_cell_type_pair",
        distance_use = TRUE,
        original_lr_pair_count = original_lr_pair_count,
        valid_lr_pair_count = nrow(lr_pairs),
        overexpressed_lr_pair_count = 0,
        zero_score_reason = "CellChat identified no overexpressed LR interaction",
        cell_count = ncol(counts),
        cell_type_resolution = metadata_resolution,
        lr_candidate_source = if (file.exists(file.path(case_dir, "LR_candidate_universe.tsv"))) "case_LR_candidate_universe" else "legacy_global_LR"
      ),
      file.path(output_dir, "manifest.json"),
      pretty = TRUE, auto_unbox = TRUE
    )
    return(invisible(NULL))
  }
  positive_distances <- as.numeric(dist(coordinates))
  interaction_range <- as.numeric(quantile(positive_distances[positive_distances > 0], 0.25))
  object <- computeCommunProb(
    object,
    type = "truncatedMean",
    trim = 0.1,
    distance.use = TRUE,
    interaction.range = interaction_range,
    # The simulator contains unit-spaced grid neighbours.  A scale of 1 keeps
    # CellChat's minimum scaled group distance in its recommended [1, 2]
    # interval; the legacy value 0.01 is invalid for these coordinates.
    scale.distance = 1,
    contact.dependent = FALSE,
    nboot = 20,
    seed.use = seed_value
  )
  probability <- object@net$prob
  if (is.null(probability) || length(probability) == 0) stop("CellChat returned no probability array")
  group_score <- apply(probability, c(1, 2), sum)
  write.table(
    group_score,
    file.path(output_dir, "group_score.tsv"),
    sep = "\t", quote = FALSE, col.names = NA
  )
  write_lr_score(object, probability, output_dir)
  write_json(
    list(
      method = "CellChat_2.2.0",
      case = case_key,
      resolution = "directed_cell_type_pair",
      distance_use = TRUE,
      interaction_range = interaction_range,
      original_lr_pair_count = original_lr_pair_count,
      valid_lr_pair_count = nrow(lr_pairs),
      cell_count = ncol(counts),
      cell_type_resolution = metadata_resolution,
      lr_candidate_source = if (file.exists(file.path(case_dir, "LR_candidate_universe.tsv"))) "case_LR_candidate_universe" else "legacy_global_LR"
    ),
    file.path(output_dir, "manifest.json"),
    pretty = TRUE, auto_unbox = TRUE
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("usage: run_cellchat_group_native.R SIM_ROOT LR_ROOT OUTPUT_ROOT CASE_KEY [CASE_KEY ...]")
}
sim_root <- args[1]
lr_root <- args[2]
output_root <- args[3]
case_keys <- args[4:length(args)]
if (identical(case_keys[1], "--all")) {
  gene_dirs <- list.dirs(sim_root, recursive = FALSE, full.names = TRUE)
  gene_dirs <- gene_dirs[grepl("^geneNum_[0-9]+_id_[0-9]+$", basename(gene_dirs))]
  case_keys <- character()
  for (gene_dir in gene_dirs) {
    cell_type_dirs <- list.dirs(gene_dir, recursive = FALSE, full.names = TRUE)
    cell_type_dirs <- cell_type_dirs[grepl("^cellType[0-9]+$", basename(cell_type_dirs))]
    for (cell_type_dir in cell_type_dirs) {
      seed_dirs <- list.dirs(cell_type_dir, recursive = FALSE, full.names = TRUE)
      seed_dirs <- seed_dirs[grepl("^seed[0-9]+$", basename(seed_dirs))]
      seed_dirs <- seed_dirs[basename(seed_dirs) %in% c("seed12", "seed22", "seed32", "seed42", "seed52")]
      case_keys <- c(
        case_keys,
        file.path(basename(gene_dir), basename(cell_type_dir), basename(seed_dirs))
      )
    }
  }
  case_keys <- sort(case_keys)
}
shard_count <- as.integer(Sys.getenv("CELLCHAT_SHARD_COUNT", "1"))
shard_index <- as.integer(Sys.getenv("CELLCHAT_SHARD_INDEX", "0"))
if (shard_count < 1 || shard_index < 0 || shard_index >= shard_count) {
  stop("invalid CELLCHAT_SHARD_INDEX/CELLCHAT_SHARD_COUNT")
}
case_keys <- case_keys[(seq_along(case_keys) - 1) %% shard_count == shard_index]
for (position in seq_along(case_keys)) {
  case_key <- case_keys[position]
  message(sprintf("[%d/%d] %s", position, length(case_keys), case_key))
  score_path <- file.path(output_root, case_key, "group_score.tsv")
  lr_score_path <- file.path(output_root, case_key, "lr_score.tsv")
  if (file.exists(score_path) && file.exists(lr_score_path)) {
    message("  skip complete")
    next
  }
  tryCatch(
    process_case(case_key, sim_root, lr_root, output_root),
    error = function(error) {
      failure_dir <- file.path(output_root, case_key)
      dir.create(failure_dir, recursive = TRUE, showWarnings = FALSE)
      write_json(
        list(
          case = case_key,
          status = "failed",
          message = conditionMessage(error),
          call = paste(deparse(conditionCall(error)), collapse = " ")
        ),
        file.path(failure_dir, "failure.json"),
        pretty = TRUE, auto_unbox = TRUE
      )
      message(sprintf("  FAILED: %s", conditionMessage(error)))
    }
  )
}
