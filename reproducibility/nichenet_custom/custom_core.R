read_numeric_matrix <- function(path) {
  x <- as.matrix(read.delim(path, row.names = 1, check.names = FALSE))
  storage.mode(x) <- "double"
  if (anyDuplicated(rownames(x)) || anyDuplicated(colnames(x))) stop("Duplicate matrix identifiers: ", basename(path))
  if (anyNA(x) || any(!is.finite(x)) || any(x < 0)) stop("Invalid nonnegative matrix: ", basename(path))
  x
}

write_tsv <- function(x, path) write.table(x, path, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")

reconstruct_atac_gr <- function(atac, region_gene, region_tf, genes) {
  if (!setequal(rownames(atac), rownames(region_gene)) || !setequal(rownames(atac), rownames(region_tf))) stop("ATAC region labels differ")
  if (!setequal(colnames(region_gene), genes) || !all(colnames(region_tf) %in% genes)) stop("ATAC gene labels differ")
  regions <- rownames(atac)
  rg <- region_gene[regions, genes, drop = FALSE]
  rt <- region_tf[regions, , drop = FALSE]
  raw <- t(rt) %*% (rowMeans(atac) * rg)
  all_weights <- matrix(0, length(genes), length(genes), dimnames = list(genes, genes))
  all_weights[rownames(raw), genes] <- raw
  diag(all_weights) <- 0
  if (max(all_weights) > 0) all_weights <- all_weights/max(all_weights)
  all_weights
}

partial_rank_correlation <- function(normalized_expression, types) {
  ranked <- t(apply(normalized_expression, 1, rank, ties.method = "average"))
  dimnames(ranked) <- dimnames(normalized_expression)
  for (ct in unique(types)) {
    idx <- which(types == ct)
    ranked[, idx] <- ranked[, idx, drop = FALSE] - rowMeans(ranked[, idx, drop = FALSE])
  }
  norm <- sqrt(rowSums(ranked^2))
  norm[norm == 0] <- 1
  unit <- ranked/norm
  rho <- unit %*% t(unit)
  rho[rho > 1] <- 1
  rho[rho < -1] <- -1
  dimnames(rho) <- list(rownames(normalized_expression), rownames(normalized_expression))
  rho
}

build_proxy_edges <- function(rho, receptors, active_tfs) {
  result <- list()
  k <- max(1L, as.integer(ceiling(sqrt(length(active_tfs)))))
  for (receptor in sort(unique(receptors))) {
    candidates <- sort(setdiff(active_tfs, receptor))
    if (!length(candidates)) next
    weights <- abs(rho[receptor, candidates])
    selected <- head(candidates[order(-weights, candidates)], k)
    selected <- selected[abs(rho[receptor, selected]) > 1e-12]
    if (!length(selected)) next
    result[[receptor]] <- data.frame(from = receptor, to = selected, weight = as.numeric(abs(rho[receptor, selected])), signed_association = as.numeric(rho[receptor, selected]))
  }
  if (!length(result)) return(data.frame(from = character(), to = character(), weight = numeric(), signed_association = numeric()))
  do.call(rbind, result)
}

empty_lr_score <- function() data.frame(sender_type = character(), receiver_type = character(), ligand = character(), receptor = character(), score = numeric())

select_response_genes <- function(marker_table, receiver, background, fraction = 0.2) {
  if (length(background) < 2) return(character())
  selected <- marker_table[as.character(marker_table$cluster_id) == receiver & marker_table$gene %in% background, , drop = FALSE]
  score <- setNames(rep(-Inf, length(background)), background)
  score[selected$gene] <- selected$avg_log2FC
  if (all(!is.finite(score))) return(character())
  number <- min(length(background)-1L, max(1L, as.integer(ceiling(fraction*length(background)))))
  head(names(score)[order(-score, names(score))], number)
}
