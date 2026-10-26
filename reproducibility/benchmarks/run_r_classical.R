#!/usr/bin/env Rscript

# Canonical ARACNE/CLR/GENIE3 inference for simulated GRN benchmarks.
# This script writes raw score matrices only.  The shared Python evaluator is
# intentionally used afterwards so every language and method has identical
# candidate edges and metric implementations.

suppressPackageStartupMessages(library(minet))

method <- tolower(Sys.getenv("GRN_METHOD"))
if (!method %in% c("aracne", "clr", "genie3")) {
  stop("GRN_METHOD must be one of: aracne, clr, genie3")
}
if (method == "genie3") suppressPackageStartupMessages(library(GENIE3))

sim_root <- Sys.getenv("GRN_SIM_ROOT", "data/simulation/simdata")
output_root <- Sys.getenv("GRN_OUTPUT_ROOT")
limit_text <- Sys.getenv("GRN_LIMIT", "")
force <- identical(tolower(Sys.getenv("GRN_FORCE", "false")), "true")
if (output_root == "") stop("GRN_OUTPUT_ROOT is required")

read_tf_ids <- function(path) {
  frame <- read.table(path, header=TRUE, sep="\t", check.names=FALSE,
                      stringsAsFactors=FALSE, quote="", comment.char="")
  ids <- trimws(as.character(frame[[1]]))
  if (length(ids) == 0 || anyDuplicated(ids)) stop(paste("invalid TF IDs:", path))
  ids
}

read_metadata <- function(path, cells) {
  metadata <- read.table(path, header=FALSE, sep="\t", col.names=c("cell", "cell_type"),
                         stringsAsFactors=FALSE, quote="", comment.char="")
  metadata$cell <- trimws(as.character(metadata$cell))
  metadata$cell_type <- trimws(as.character(metadata$cell_type))
  if (anyDuplicated(metadata$cell) || !setequal(metadata$cell, cells)) {
    stop(paste("expression/metadata mismatch:", path))
  }
  metadata[match(cells, metadata$cell), , drop=FALSE]
}

infer_scores <- function(expression, tf_ids, method) {
  genes <- rownames(expression)
  keep <- apply(expression, 1, sd) > 1e-12
  usable <- expression[keep, , drop=FALSE]
  scores <- matrix(0, nrow=length(genes), ncol=length(genes),
                   dimnames=list(genes, genes))
  if (nrow(usable) < 2) return(scores)

  if (method %in% c("aracne", "clr")) {
    samples_by_genes <- scale(log1p(t(usable)))
    samples_by_genes[!is.finite(samples_by_genes)] <- 0
    mim <- minet::build.mim(samples_by_genes, estimator="spearman")
    inferred <- if (method == "aracne") minet::aracne(mim) else minet::clr(mim)
    scores[rownames(inferred), colnames(inferred)] <- inferred
  } else {
    valid_tfs <- intersect(tf_ids, rownames(usable))
    if (length(valid_tfs) == 0) stop("no variable TF is available to GENIE3")
    inferred <- GENIE3::GENIE3(
      as.matrix(usable), regulators=valid_tfs, nCores=1, verbose=FALSE
    )
    scores[rownames(inferred), colnames(inferred)] <- inferred
  }
  diag(scores) <- 0
  if (any(!is.finite(scores))) stop("inference produced non-finite scores")
  scores
}

dataset_dirs <- list.dirs(sim_root, recursive=FALSE, full.names=TRUE)
dataset_dirs <- dataset_dirs[grepl("^geneNum_[0-9]+_id_[0-9]+$", basename(dataset_dirs))]
dataset_parts <- do.call(rbind, lapply(dataset_dirs, function(path) {
  hits <- regmatches(basename(path), regexec("^geneNum_([0-9]+)_id_([0-9]+)$", basename(path)))[[1]]
  c(path=path, gene_num=hits[2], dataset_id=hits[3])
}))
ordering <- order(as.integer(dataset_parts[, "dataset_id"]))
dataset_parts <- dataset_parts[ordering, , drop=FALSE]

cases <- list()
for (row in seq_len(nrow(dataset_parts))) {
  for (cell_type_count in c(2, 3, 4)) {
    for (seed in c(12, 22, 32, 42, 52)) {
      data_dir <- file.path(dataset_parts[row, "path"], paste0("cellType", cell_type_count), paste0("seed", seed))
      if (!dir.exists(data_dir)) stop(paste("missing canonical case:", data_dir))
      cases[[length(cases) + 1]] <- list(
        gene_num=dataset_parts[row, "gene_num"],
        dataset_id=dataset_parts[row, "dataset_id"],
        cell_type_count=cell_type_count,
        seed=seed,
        data_dir=data_dir
      )
    }
  }
}
if (limit_text != "") cases <- cases[seq_len(min(as.integer(limit_text), length(cases)))]

manifest_dir <- file.path(output_root, method)
dir.create(manifest_dir, recursive=TRUE, showWarnings=FALSE)
writeLines(c(
  paste0("method\t", method),
  paste0("sim_root\t", normalizePath(sim_root)),
  paste0("case_count\t", length(cases)),
  paste0("pid\t", Sys.getpid())
), file.path(manifest_dir, "manifest.tsv"))

for (position in seq_along(cases)) {
  case <- cases[[position]]
  case_key <- file.path(
    paste0("geneNum_", case$gene_num, "_id_", case$dataset_id),
    paste0("cellType", case$cell_type_count), paste0("seed", case$seed)
  )
  done_path <- file.path(output_root, method, case_key, "INFERENCE_COMPLETE")
  if (file.exists(done_path) && !force) {
    message(sprintf("[%d/%d] skip complete %s", position, length(cases), case_key))
    next
  }
  message(sprintf("[%d/%d] %s %s", position, length(cases), method, case_key))
  expression <- as.matrix(read.table(
    file.path(case$data_dir, "rnaseq_data.txt"), header=TRUE, sep="\t",
    row.names=1, check.names=FALSE
  ))
  rownames(expression) <- trimws(as.character(rownames(expression)))
  colnames(expression) <- trimws(as.character(colnames(expression)))
  metadata <- read_metadata(file.path(case$data_dir, "metadata.txt"), colnames(expression))
  tf_ids <- read_tf_ids(file.path(case$data_dir, "tf_ids.txt"))
  observed_types <- unique(metadata$cell_type)
  if (length(observed_types) != case$cell_type_count) stop(paste("cell-type count mismatch:", case_key))

  for (cell_type in observed_types) {
    cells <- metadata$cell[metadata$cell_type == cell_type]
    if (length(cells) < 5) stop(paste("fewer than five cells:", case_key, cell_type))
    scores <- infer_scores(expression[, cells, drop=FALSE], tf_ids, method)
    score_dir <- file.path(output_root, method, case_key, paste0("cell_type=", cell_type))
    dir.create(score_dir, recursive=TRUE, showWarnings=FALSE)
    temporary <- file.path(score_dir, "score.tsv.tmp")
    write.table(scores, temporary, sep="\t", quote=FALSE, col.names=NA)
    if (!file.rename(temporary, file.path(score_dir, "score.tsv"))) stop("atomic score rename failed")
  }
  writeLines("ok", done_path)
}
