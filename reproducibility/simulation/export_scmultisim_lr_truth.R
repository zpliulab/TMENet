#!/usr/bin/env Rscript

# Export direction-correct, fine-cell-state LR truth from saved scMultiSim
# objects without overwriting the legacy simulation files.

export_case <- function(sim_root, output_root, case_key) {
  case_dir <- file.path(sim_root, case_key)
  object <- readRDS(file.path(case_dir, "data.rds"))
  coarse <- read.table(
    file.path(case_dir, "metadata.txt"), sep = "\t", header = FALSE,
    col.names = c("cell", "coarse_cell_type"), stringsAsFactors = FALSE
  )
  if (nrow(coarse) != nrow(object$cell_meta)) {
    stop(sprintf("cell metadata length mismatch: %s", case_key))
  }
  metadata <- data.frame(
    cell = as.character(coarse$cell),
    cell_type = as.character(object$cell_meta$cell.type),
    cell_type_index = as.integer(object$cell_meta$cell.type.idx),
    coarse_cell_type = as.character(coarse$coarse_cell_type),
    stringsAsFactors = FALSE
  )
  if (anyDuplicated(metadata$cell)) stop("duplicate cell identities")

  params <- object$cci_cell_type_param
  edge_parts <- vector("list", nrow(params))
  for (row_index in seq_len(nrow(params))) {
    # .rnaSeqSpatial uses sp_ctype_param[receiver, neighbour_sender, lr].
    receiver_type <- params$ct1[row_index]
    sender_type <- params$ct2[row_index]
    receiver_positions <- which(object$cell_meta$cell.type.idx == receiver_type)
    rows <- list()
    part_index <- 0L
    for (receiver_position in receiver_positions) {
      neighbours <- object$grid$get_neighbours(receiver_position)
      senders <- neighbours[
        !is.na(neighbours) & object$cell_meta$cell.type.idx[neighbours] == sender_type
      ]
      if (length(senders)) {
        part_index <- part_index + 1L
        rows[[part_index]] <- data.frame(
          sender = metadata$cell[senders],
          receiver = metadata$cell[receiver_position],
          ligand = as.character(params$ligand[row_index]),
          receptor = as.character(params$receptor[row_index]),
          sender_type = metadata$cell_type[senders],
          receiver_type = metadata$cell_type[receiver_position],
          stringsAsFactors = FALSE
        )
      }
    }
    if (length(rows)) edge_parts[[row_index]] <- do.call(rbind, rows)
  }
  edge_parts <- Filter(Negate(is.null), edge_parts)
  edges <- if (length(edge_parts)) do.call(rbind, edge_parts) else data.frame(
    sender = character(), receiver = character(), ligand = character(),
    receptor = character(), sender_type = character(), receiver_type = character()
  )
  edges <- unique(edges)

  output_dir <- file.path(output_root, case_key)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write.table(
    metadata, file.path(output_dir, "cell_metadata.tsv"), sep = "\t",
    quote = FALSE, row.names = FALSE
  )
  write.table(
    edges, file.path(output_dir, "realized_lr_edges.tsv"), sep = "\t",
    quote = FALSE, row.names = FALSE
  )
  manifest <- list(
    case = case_key,
    direction = "neighbour_sender_ligand_to_current_receiver_receptor",
    cell_type_resolution = "scMultiSim_cell_meta_cell.type",
    coarse_cell_type_count = length(unique(metadata$coarse_cell_type)),
    fine_cell_type_count = length(unique(metadata$cell_type)),
    realized_unique_cell_lr_edges = nrow(edges)
  )
  writeLines(jsonlite::toJSON(manifest, pretty = TRUE, auto_unbox = TRUE), file.path(output_dir, "manifest.json"))
  message(sprintf("%s: %d fine types, %d realized LR edges", case_key, manifest$fine_cell_type_count, nrow(edges)))
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: export_scmultisim_lr_truth.R SIM_ROOT OUTPUT_ROOT CASE_KEY [CASE_KEY ...]")
}
sim_root <- args[1]
output_root <- args[2]
for (case_key in args[3:length(args)]) export_case(sim_root, output_root, case_key)
