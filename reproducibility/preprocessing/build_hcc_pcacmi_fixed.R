#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(infotheo))

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  values <- list(
    input_dir = "data/HCC",
    output_dir = "results/hcc_revision_20260915/pcacmi_fixed_v1",
    lambda = 0.03,
    max_order = 2L
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (i == length(args)) stop(paste("Missing value for", key))
    value <- args[[i + 1L]]
    if (key == "--input-dir") values$input_dir <- value
    else if (key == "--output-dir") values$output_dir <- value
    else if (key == "--lambda") values$lambda <- as.numeric(value)
    else if (key == "--max-order") values$max_order <- as.integer(value)
    else stop(paste("Unknown argument:", key))
    i <- i + 2L
  }
  if (!is.finite(values$lambda) || values$lambda < 0) stop("lambda must be nonnegative")
  if (is.na(values$max_order) || values$max_order < 0L || values$max_order > 3L) {
    stop("max-order must be an integer in [0,3]")
  }
  values
}

discretize_expression <- function(data) {
  if (!is.matrix(data) || nrow(data) < 2L || ncol(data) < 2L) {
    stop("Expression input must be a genes-by-samples matrix with at least two rows and columns")
  }
  if (any(!is.finite(data))) stop("Expression matrix contains non-finite values")
  nbins <- max(2L, as.integer(round(sqrt(ncol(data)))))
  discrete <- t(discretize(t(data), disc = "equalwidth", nbins = nbins))
  storage.mode(discrete) <- "integer"
  discrete
}

reduce_order <- function(G, discrete, order, lambda) {
  n <- nrow(G)
  if (!identical(dim(G), c(n, n))) stop("G must be square")
  skeleton <- (G != 0) | (t(G) != 0)
  diag(skeleton) <- FALSE
  candidate_pairs <- which(upper.tri(skeleton) & skeleton, arr.ind = TRUE)
  checked <- 0L
  removed <- 0L

  if (nrow(candidate_pairs) == 0L) {
    return(list(G = G, checked_pairs = checked, removed_directed_edges = removed))
  }

  for (pair_index in seq_len(nrow(candidate_pairs))) {
    i <- candidate_pairs[pair_index, 1L]
    j <- candidate_pairs[pair_index, 2L]
    score <- NA_real_

    if (order == 0L) {
      score <- mutinformation(
        as.numeric(discrete[i, ]), as.numeric(discrete[j, ]), method = "emp"
      )
    } else {
      conditioning <- which(skeleton[i, ] & skeleton[j, ])
      conditioning <- setdiff(conditioning, c(i, j))
      if (length(conditioning) < order) next
      combinations <- combn(conditioning, order, simplify = FALSE)
      conditional_scores <- vapply(combinations, function(indices) {
        value <- condinformation(
          as.numeric(discrete[i, ]),
          as.numeric(discrete[j, ]),
          as.data.frame(t(discrete[indices, , drop = FALSE])),
          method = "emp"
        )
        if (is.nan(value) || !is.finite(value)) 0 else value
      }, numeric(1L))
      score <- max(conditional_scores)
    }

    checked <- checked + 1L
    if (is.nan(score) || !is.finite(score)) score <- 0
    if (score < lambda) {
      before <- as.integer(G[i, j] != 0) + as.integer(G[j, i] != 0)
      # Association evidence is symmetric, but the database direction is preserved:
      # weak evidence removes existing directions and never creates the reverse edge.
      G[i, j] <- 0
      G[j, i] <- 0
      removed <- removed + before
    }
  }
  list(G = G, checked_pairs = checked, removed_directed_edges = removed)
}

pca_cmi_fixed <- function(data, lambda = 0.03, max_order = 2L, G) {
  if (!is.matrix(G) || nrow(G) != ncol(G) || nrow(G) != nrow(data)) {
    stop("G must be square and aligned to expression rows")
  }
  G <- 1L * (G > 0)
  diag(G) <- 0L
  discrete <- discretize_expression(data)
  history <- list()

  # The legacy implementation incremented order before this call, so order 0
  # was unreachable.  Iterate explicitly from zero.
  for (order in 0:max_order) {
    result <- reduce_order(G, discrete, order, lambda)
    G <- result$G
    history[[length(history) + 1L]] <- data.frame(
      order = order,
      checked_pairs = result$checked_pairs,
      removed_directed_edges = result$removed_directed_edges,
      remaining_directed_edges = sum(G != 0),
      stringsAsFactors = FALSE
    )
    message(sprintf(
      "order=%d checked_pairs=%d removed_directed_edges=%d remaining=%d",
      order, result$checked_pairs, result$removed_directed_edges, sum(G != 0)
    ))
    if (order > 0L && result$checked_pairs == 0L) break
  }
  list(G = G, history = do.call(rbind, history), discrete = discrete)
}

write_matrix <- function(matrix, path) {
  write.table(matrix, file = path, sep = "\t", quote = FALSE, col.names = NA)
}

main <- function() {
  opts <- parse_args()
  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  prior_path <- file.path(opts$input_dir, "GRN_all.txt")
  metadata_path <- file.path(opts$input_dir, "metadata.txt")
  if (!file.exists(prior_path) || !file.exists(metadata_path)) stop("Missing HCC prior or metadata")

  prior <- as.matrix(read.table(
    prior_path, header = TRUE, row.names = 1, sep = "\t", check.names = FALSE
  ))
  if (!identical(rownames(prior), colnames(prior))) stop("GRN_all row/column labels differ")
  metadata <- read.table(metadata_path, header = FALSE, sep = "\t", stringsAsFactors = FALSE)
  cell_types <- sort(unique(as.character(metadata[, 2L])))
  expected <- c("Fibroblast", "Normal_Hepatocyte", "SPP1_Macrophage", "Stromal_Other", "Tumor")
  if (!identical(cell_types, expected)) stop("Unexpected HCC cell-type order")

  matrices <- list()
  histories <- list()
  manifest_rows <- list()
  for (cell_type in cell_types) {
    expression_path <- file.path(opts$input_dir, paste0("data_", cell_type, ".txt"))
    expression <- as.matrix(read.table(
      expression_path, header = TRUE, row.names = 1, sep = "\t", check.names = FALSE
    ))
    if (!setequal(rownames(prior), rownames(expression))) {
      stop(paste("Gene sets differ for", cell_type))
    }
    expression <- expression[rownames(prior), , drop = FALSE]
    message(sprintf("cell_type=%s genes=%d samples=%d", cell_type, nrow(expression), ncol(expression)))
    result <- pca_cmi_fixed(expression, opts$lambda, opts$max_order, prior)
    rownames(result$G) <- rownames(prior)
    colnames(result$G) <- colnames(prior)
    matrices[[cell_type]] <- result$G
    history <- result$history
    history$cell_type <- cell_type
    histories[[cell_type]] <- history[, c("cell_type", "order", "checked_pairs", "removed_directed_edges", "remaining_directed_edges")]
    manifest_rows[[cell_type]] <- data.frame(
      cell_type = cell_type,
      genes = nrow(expression),
      samples = ncol(expression),
      input_directed_edges = sum(prior != 0),
      output_directed_edges = sum(result$G != 0),
      stringsAsFactors = FALSE
    )
    write_matrix(result$G, file.path(opts$output_dir, paste0("GRN_Refined_", cell_type, ".txt")))
  }

  full <- as.matrix(Matrix::bdiag(matrices))
  labels <- unlist(lapply(matrices, rownames), use.names = FALSE)
  rownames(full) <- labels
  colnames(full) <- labels
  write_matrix(full, file.path(opts$output_dir, "B_star.txt"))
  write.table(do.call(rbind, histories), file.path(opts$output_dir, "pcacmi_order_history.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(do.call(rbind, manifest_rows), file.path(opts$output_dir, "pcacmi_cell_summary.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  writeLines(c(
    paste0("input_dir=", normalizePath(opts$input_dir)),
    paste0("lambda=", opts$lambda),
    paste0("max_order=", opts$max_order),
    "orientation=source_rows_target_columns",
    "direction_policy=preserve_database_directions_never_create_reverse_edges",
    "order_policy=explicit_zero_through_max_order"
  ), file.path(opts$output_dir, "RUN_INFO.txt"))
  message(paste("Completed fixed HCC PCA-CMI:", opts$output_dir))
}

if (sys.nframe() == 0L) main()
