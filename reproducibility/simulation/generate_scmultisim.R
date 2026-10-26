#!/usr/bin/env Rscript

# Reproducible, resume-safe scMultiSim generation for TMENet simulations.
# Existing source networks are read-only and outputs go to a versioned root.

suppressPackageStartupMessages({
  library(scMultiSim)
  library(ape)
  library(jsonlite)
})

canonicalize_edges <- function(path, allow_duplicates = FALSE) {
  edges <- read.table(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
  if (!identical(colnames(edges), c("target", "regulator", "effect"))) {
    stop(sprintf("unexpected columns: %s", path))
  }
  edges$target <- as.integer(edges$target)
  edges$regulator <- as.integer(edges$regulator)
  edges$effect <- as.numeric(edges$effect)
  if (any(!is.finite(edges$effect))) stop("non-finite edge effect")
  key <- paste(edges$target, edges$regulator, sep = "->")
  duplicate_count <- sum(duplicated(key))
  if (duplicate_count && !allow_duplicates) stop("duplicate GRN edges")
  if (duplicate_count) edges <- edges[!duplicated(key, fromLast = TRUE), , drop = FALSE]
  attr(edges, "input_rows") <- length(key)
  attr(edges, "duplicate_rows") <- duplicate_count
  edges
}

read_candidate_lr <- function(path, true_cci, gene_num) {
  if (!file.exists(path)) stop(sprintf("missing LR candidate universe: %s", path))
  candidates <- read.table(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
  if (!identical(colnames(candidates), c("ligand", "receptor", "truth", "effect"))) {
    stop(sprintf("unexpected candidate LR columns: %s", path))
  }
  candidates$ligand <- as.integer(candidates$ligand)
  candidates$receptor <- as.integer(candidates$receptor)
  candidates$truth <- as.integer(candidates$truth)
  candidates$effect <- as.numeric(candidates$effect)
  if (any(!is.finite(candidates$effect))) stop("non-finite candidate LR effect")
  if (any(!candidates$truth %in% c(0L, 1L))) stop("candidate LR truth must be binary")
  if (any(candidates$ligand < 1L | candidates$receptor < 1L |
          candidates$ligand > gene_num | candidates$receptor > gene_num)) {
    stop("candidate LR gene index is out of bounds")
  }
  candidate_key <- paste(candidates$receptor, candidates$ligand, sep = "->")
  if (anyDuplicated(candidate_key)) stop("duplicate candidate LR pairs")
  if (any(candidates$ligand == candidates$receptor)) stop("self candidate LR pair")
  true_candidates <- candidates[candidates$truth == 1L, , drop = FALSE]
  candidate_true_key <- paste(true_candidates$receptor, true_candidates$ligand, sep = "->")
  cci_key <- paste(true_cci$target, true_cci$regulator, sep = "->")
  if (!setequal(candidate_true_key, cci_key)) stop("candidate truth does not match injected CCI")
  candidate_effect <- setNames(true_candidates$effect, candidate_true_key)
  cci_effect <- setNames(true_cci$effect, cci_key)
  if (any(abs(candidate_effect[cci_key] - cci_effect) > 1e-12)) {
    stop("candidate truth effects do not match injected CCI")
  }
  candidates
}

tree_for_tips <- function(count) {
  labels <- LETTERS[seq_len(count)]
  read.tree(text = sprintf("(%s);", paste0(labels, ":1", collapse = ",")))
}

write_matrix <- function(matrix_value, path) {
  write.table(matrix_value, path, sep = "\t", quote = FALSE, col.names = NA)
}

tie_inclusive_knn <- function(distance, k = 4L) {
  safe <- distance
  diag(safe) <- Inf
  k_eff <- min(as.integer(k), nrow(safe) - 1L)
  kth <- apply(safe, 1, function(row) sort(row, partial = k_eff)[k_eff])
  mask <- safe <= kth
  diag(mask) <- FALSE
  mask
}

build_a <- function(expression, coarse_metadata) {
  types <- sort(unique(coarse_metadata$cell_type))
  ordered_cells <- unlist(lapply(types, function(type) coarse_metadata$cell[coarse_metadata$cell_type == type]))
  blocks <- lapply(types, function(type) expression[, coarse_metadata$cell[coarse_metadata$cell_type == type], drop = FALSE])
  result <- matrix(0, nrow = nrow(expression) * length(types), ncol = ncol(expression))
  rownames(result) <- unlist(lapply(types, function(type) paste(type, rownames(expression), sep = "_")))
  colnames(result) <- ordered_cells
  row_offset <- 0L
  col_offset <- 0L
  for (block in blocks) {
    result[row_offset + seq_len(nrow(block)), col_offset + seq_len(ncol(block))] <- as.matrix(block)
    row_offset <- row_offset + nrow(block)
    col_offset <- col_offset + ncol(block)
  }
  result
}

exact_lr_truth <- function(result, cells, effective_cci) {
  gt <- result$cci_gt
  if (is.null(gt) || length(dim(gt)) != 3L) stop("single-cell CCI truth is unavailable")
  regulators <- as.integer(result$.n$sp_regulators)
  # .n stores a count; the ordered identities are retained in options params.
  regulator_ids <- sort(unique(as.integer(effective_cci$regulator)))
  if (dim(gt)[3] != length(regulator_ids)) stop("CCI truth regulator dimension mismatch")
  edge_parts <- list()
  part_index <- 0L
  for (receiver in seq_along(cells)) {
    neighbours <- result$grid$get_neighbours(receiver)
    neighbours <- neighbours[!is.na(neighbours)]
    for (sender in neighbours) {
      active_indices <- which(gt[receiver, sender, ] > 0)
      for (regulator_index in active_indices) {
        ligand <- regulator_ids[regulator_index]
        receptor_rows <- effective_cci[effective_cci$regulator == ligand, , drop = FALSE]
        if (nrow(receptor_rows)) {
          part_index <- part_index + 1L
          edge_parts[[part_index]] <- data.frame(
            sender = cells[sender], receiver = cells[receiver],
            ligand = as.character(ligand), receptor = as.character(receptor_rows$target),
            effect = receptor_rows$effect, stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  if (!length(edge_parts)) stop("no exact LR truth edges were generated")
  unique(do.call(rbind, edge_parts))
}

build_ccc_prior <- function(expression, effective_cci, distance, k = 4L) {
  cells <- colnames(expression)
  support <- matrix(FALSE, length(cells), length(cells), dimnames = list(cells, cells))
  weighted <- matrix(0, length(cells), length(cells), dimnames = list(cells, cells))
  for (row_index in seq_len(nrow(effective_cci))) {
    ligand <- as.character(effective_cci$regulator[row_index])
    receptor <- as.character(effective_cci$target[row_index])
    ligand_values <- as.numeric(expression[ligand, ])
    receptor_values <- as.numeric(expression[receptor, ])
    support <- support | outer(ligand_values > 0, receptor_values > 0, "&")
    weighted <- weighted + outer(ligand_values, receptor_values, "*")
  }
  mask <- tie_inclusive_knn(distance, k)
  binary <- (support & mask) * 1L
  weighted <- weighted * mask
  if (max(weighted) > 0) weighted <- weighted / max(weighted)
  diag(binary) <- 0L
  diag(weighted) <- 0
  list(binary = binary, weighted = weighted, mask = mask)
}

process_case <- function(grn_path, cci_path, output_root, gene_num, dataset_id, cell_type_count, seed) {
  case_key <- sprintf("geneNum_%d_id_%d/cellType%d/seed%d", gene_num, dataset_id, cell_type_count, seed)
  output_dir <- file.path(output_root, case_key)
  complete_path <- file.path(output_dir, "COMPLETE.json")
  if (file.exists(complete_path)) {
    message(sprintf("skip complete %s", case_key))
    return(invisible(NULL))
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  started <- Sys.time()
  grn <- canonicalize_edges(grn_path, allow_duplicates = FALSE)
  cci <- canonicalize_edges(cci_path, allow_duplicates = TRUE)
  if (max(grn$target, grn$regulator, cci$target, cci$regulator) > gene_num) stop("gene index exceeds filename gene_num")
  candidate_source <- file.path(dirname(cci_path), sprintf("candidate_lr_pairs_%d_%d.tsv", gene_num, dataset_id))
  candidates <- read_candidate_lr(candidate_source, cci, gene_num)
  candidate_edges <- data.frame(
    target = candidates$receptor,
    regulator = candidates$ligand,
    effect = candidates$effect,
    stringsAsFactors = FALSE
  )
  algorithm_lr_input <- candidates[, c("ligand", "receptor"), drop = FALSE]
  dataset_dir <- file.path(output_root, sprintf("geneNum_%d_id_%d", gene_num, dataset_id))
  dir.create(dataset_dir, recursive = TRUE, showWarnings = FALSE)
  write.table(grn, file.path(dataset_dir, "GRN_truth_edges.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(cci, file.path(dataset_dir, "CCI_truth_edges.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  grn_truth <- matrix(0L, gene_num, gene_num, dimnames = list(as.character(seq_len(gene_num)), as.character(seq_len(gene_num))))
  grn_truth[cbind(grn$regulator, grn$target)] <- 1L
  write_matrix(grn_truth, file.path(dataset_dir, "gold_GRN.txt"))
  file.copy(candidate_source, file.path(dataset_dir, "LR_candidate_universe.tsv"), overwrite = TRUE)
  write.table(
    algorithm_lr_input, file.path(dataset_dir, "LR_candidates_input.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
  )

  tree <- tree_for_tips(cell_type_count)
  options <- list(
    rand.seed = seed,
    GRN = grn,
    num.cells = cell_type_count * 50L,
    num.cifs = 20L,
    cif.sigma = 0.1,
    diff.cif.fraction = 0.9,
    do.velocity = TRUE,
    tree = tree,
    cci = list(
      params = cci,
      max.neighbors = 4L,
      grid.size = cell_type_count * 50L + 300L,
      layout = "enhanced",
      step.size = 0.5,
      cell.type.interaction = "random",
      single.cell.gt = TRUE
    )
  )
  set.seed(seed)
  result <- sim_true_counts(options)
  cells <- sub("^cell", "", colnames(result$counts))
  expression <- as.matrix(result$counts[seq_len(gene_num), , drop = FALSE])
  colnames(expression) <- cells
  rownames(expression) <- as.character(seq_len(gene_num))
  coarse_metadata <- data.frame(
    cell = cells, cell_type = as.character(result$cell_meta$pop), stringsAsFactors = FALSE
  )
  fine_metadata <- data.frame(
    cell = cells,
    cell_type = as.character(result$cell_meta$cell.type),
    cell_type_index = as.integer(result$cell_meta$cell.type.idx),
    coarse_cell_type = as.character(result$cell_meta$pop),
    stringsAsFactors = FALSE
  )
  coordinates <- as.matrix(result$cci_locs)
  rownames(coordinates) <- cells
  colnames(coordinates) <- c("x", "y")
  distance <- as.matrix(dist(coordinates))
  rownames(distance) <- colnames(distance) <- cells
  lr_truth <- exact_lr_truth(result, cells, cci)
  lr_truth$sender_type <- fine_metadata$cell_type[match(lr_truth$sender, fine_metadata$cell)]
  lr_truth$receiver_type <- fine_metadata$cell_type[match(lr_truth$receiver, fine_metadata$cell)]
  gold <- matrix(0L, length(cells), length(cells), dimnames = list(cells, cells))
  gold[cbind(match(lr_truth$sender, cells), match(lr_truth$receiver, cells))] <- 1L
  # Only true CCI rows are injected into scMultiSim.  The algorithm-facing
  # prior must instead use the full LR candidate universe so that LR truth is
  # not leaked into TMENet before benchmarking.
  prior <- build_ccc_prior(expression, candidate_edges, distance, 4L)
  if (any(gold != 0 & !prior$mask)) stop("gold CCC contains edges outside tie-inclusive KNN4")
  a_matrix <- build_a(expression, coarse_metadata)
  a_cells <- colnames(a_matrix)

  save_rds <- tolower(Sys.getenv("SCMULTISIM_SAVE_RDS", "false")) %in% c("1", "true", "yes")
  if (save_rds) saveRDS(result, file.path(output_dir, "data.rds"), compress = TRUE)
  write_matrix(expression, file.path(output_dir, "rnaseq_data.txt"))
  write.table(coarse_metadata, file.path(output_dir, "metadata.txt"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
  write.table(fine_metadata, file.path(output_dir, "metadata_fine.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  write.csv(coordinates, file.path(output_dir, "coords.csv"), quote = FALSE)
  write_matrix(distance, file.path(output_dir, "D.txt"))
  write_matrix(a_matrix, file.path(output_dir, "A.txt"))
  write_matrix(gold, file.path(output_dir, "gold_CCC.txt"))
  write_matrix(prior$binary, file.path(output_dir, "C_prior_knn4_binary.txt"))
  write_matrix(prior$weighted, file.path(output_dir, "C_prior_knn4_weighted.txt"))
  # TMENet consumes C_star numerically beside A, so C_star must follow A's
  # type-grouped column order even though all matrices retain explicit labels.
  write_matrix(prior$binary[a_cells, a_cells, drop = FALSE], file.path(output_dir, "C_star.txt"))
  write_matrix(grn_truth, file.path(output_dir, "gold_GRN.txt"))
  write.table(lr_truth, file.path(output_dir, "realized_lr_edges.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  # Preserve the legacy CCI truth location and schema expected by existing
  # downstream scripts: cellType*/CCI/seed*_cci.txt.
  legacy_cci_dir <- file.path(dirname(output_dir), "CCI")
  dir.create(legacy_cci_dir, recursive = TRUE, showWarnings = FALSE)
  legacy_cci <- lr_truth[, c("sender", "receiver", "ligand", "receptor", "sender_type", "receiver_type")]
  write.table(
    legacy_cci, file.path(legacy_cci_dir, sprintf("seed%d_cci.txt", seed)),
    sep = "\t", quote = FALSE, row.names = FALSE
  )
  write.table(cci, file.path(output_dir, "effective_cci_input.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(candidates, file.path(output_dir, "LR_candidate_universe.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(
    algorithm_lr_input, file.path(output_dir, "LR_candidates_input.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
  )

  atac_rows <- min(3L * gene_num, nrow(result$atac_counts))
  region_ids <- as.character(seq_len(atac_rows))
  atac_output <- result$atac_counts[seq_len(atac_rows), , drop = FALSE]
  rownames(atac_output) <- region_ids
  colnames(atac_output) <- cells
  region_tf_output <- result$region_to_tf[seq_len(atac_rows), , drop = FALSE]
  rownames(region_tf_output) <- region_ids
  colnames(region_tf_output) <- as.character(result$.grn$regulators)
  region_gene_output <- result$region_to_gene[seq_len(atac_rows), seq_len(gene_num), drop = FALSE]
  rownames(region_gene_output) <- region_ids
  colnames(region_gene_output) <- as.character(seq_len(gene_num))
  write_matrix(atac_output, file.path(output_dir, "atac_data.txt"))
  write_matrix(region_tf_output, file.path(output_dir, "atac_region2tf.txt"))
  write_matrix(region_gene_output, file.path(output_dir, "atac_region2gene.txt"))
  tf_effect <- result$.grn$geff[seq_len(gene_num), , drop = FALSE]
  rownames(tf_effect) <- as.character(seq_len(gene_num))
  colnames(tf_effect) <- as.character(result$.grn$regulators)
  write_matrix(tf_effect, file.path(output_dir, "tf.txt"))
  write.csv(expression, file.path(output_dir, "sim_counts.csv"), quote = FALSE)
  tf_ids_legacy <- data.frame(as.character(result$.grn$regulators), stringsAsFactors = FALSE)
  colnames(tf_ids_legacy) <- "Unnamed: 0"
  write.table(
    tf_ids_legacy, file.path(output_dir, "tf_ids.txt"), sep = "\t",
    quote = FALSE, row.names = FALSE, col.names = TRUE
  )

  manifest <- list(
    case = case_key,
    generator = "generate_scmultisim_v2.R",
    scMultiSim_version = as.character(packageVersion("scMultiSim")),
    seed = seed,
    gene_num = gene_num,
    dataset_id = dataset_id,
    requested_coarse_cell_types = cell_type_count,
    realized_fine_cell_types = length(unique(fine_metadata$cell_type)),
    cell_count = length(cells),
    grn_rows = nrow(grn),
    cci_input_rows = attr(cci, "input_rows"),
    cci_duplicate_rows_removed = attr(cci, "duplicate_rows"),
    effective_cci_rows = nrow(cci),
    lr_candidate_rows = nrow(candidates),
    lr_candidate_positive_rows = sum(candidates$truth == 1L),
    ccc_prior_source = "full_LR_candidate_universe_without_truth_labels",
    exact_realized_lr_edges = nrow(lr_truth),
    gold_ccc_edges = sum(gold != 0),
    knn_candidate_edges = sum(prior$mask),
    intermediate_rds_saved = save_rds,
    grn_md5 = unname(tools::md5sum(grn_path)),
    cci_md5 = unname(tools::md5sum(cci_path)),
    started = format(started, tz = "UTC", usetz = TRUE),
    completed = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
  writeLines(toJSON(manifest, pretty = TRUE, auto_unbox = TRUE), complete_path)
  message(sprintf("complete %s: %d LR edges, %d CCC edges", case_key, nrow(lr_truth), sum(gold)))
  rm(result)
  gc(full = TRUE)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) {
  stop("usage: generate_scmultisim_v2.R GRN_ROOT CCI_ROOT OUTPUT_ROOT [SHARD_COUNT SHARD_INDEX]")
}
grn_root <- args[1]
cci_root <- args[2]
output_root <- args[3]
shard_count <- if (length(args) >= 4L) as.integer(args[4]) else 1L
shard_index <- if (length(args) >= 5L) as.integer(args[5]) else 0L
if (shard_count < 1L || shard_index < 0L || shard_index >= shard_count) stop("invalid shard")
seeds <- as.integer(strsplit(Sys.getenv("SCMULTISIM_SEEDS", "12,22,32,42,52"), ",")[[1]])
cell_type_counts <- as.integer(strsplit(Sys.getenv("SCMULTISIM_CELL_TYPES", "2,3,4"), ",")[[1]])

paths <- list.files(grn_root, pattern = "^geneNum_[0-9]+_[0-9]+\\.txt$", full.names = TRUE)
parsed <- do.call(rbind, lapply(paths, function(path) {
  match <- regexec("^geneNum_([0-9]+)_([0-9]+)\\.txt$", basename(path))
  values <- regmatches(basename(path), match)[[1]]
  data.frame(path = path, gene_num = as.integer(values[2]), dataset_id = as.integer(values[3]))
}))
parsed <- parsed[order(parsed$dataset_id), ]
parsed <- parsed[(seq_len(nrow(parsed)) - 1L) %% shard_count == shard_index, , drop = FALSE]
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

for (network_index in seq_len(nrow(parsed))) {
  network <- parsed[network_index, ]
  cci_path <- file.path(cci_root, sprintf("geneNum_%d_%d.txt", network$gene_num, network$dataset_id))
  if (!file.exists(cci_path)) stop(sprintf("missing CCI network: %s", cci_path))
  for (cell_type_count in cell_type_counts) {
    for (seed in seeds) {
      tryCatch(
        process_case(network$path, cci_path, output_root, network$gene_num, network$dataset_id, cell_type_count, seed),
        error = function(error) {
          case_key <- sprintf("geneNum_%d_id_%d/cellType%d/seed%d", network$gene_num, network$dataset_id, cell_type_count, seed)
          failure_dir <- file.path(output_root, case_key)
          dir.create(failure_dir, recursive = TRUE, showWarnings = FALSE)
          writeLines(toJSON(list(case = case_key, error = conditionMessage(error)), pretty = TRUE, auto_unbox = TRUE), file.path(failure_dir, "FAILURE.json"))
          message(sprintf("FAILED %s: %s", case_key, conditionMessage(error)))
        }
      )
    }
  }
}
