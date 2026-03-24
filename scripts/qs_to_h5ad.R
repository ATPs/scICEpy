#!/usr/bin/env Rscript

parse_args <- function(args) {
  parsed <- list(
    input_qs = NULL,
    output_h5ad = NULL,
    graph_name = NULL,
    assay = NULL,
    python = "/data/p/anaconda3/bin/python"
  )

  idx <- 1L
  while (idx <= length(args)) {
    key <- args[[idx]]
    if (!startsWith(key, "--")) {
      stop("Unexpected argument: ", key, call. = FALSE)
    }
    if (idx == length(args)) {
      stop("Missing value for argument: ", key, call. = FALSE)
    }
    value <- args[[idx + 1L]]
    key_name <- substring(key, 3L)
    if (!key_name %in% names(parsed)) {
      stop("Unknown argument: ", key, call. = FALSE)
    }
    parsed[[key_name]] <- value
    idx <- idx + 2L
  }

  if (is.null(parsed$input_qs) || is.null(parsed$output_h5ad)) {
    stop(
      paste(
        "Usage:",
        "qs_to_h5ad.R --input_qs <input.qs> --output_h5ad <output.h5ad>",
        "[--graph_name <graph>] [--assay <assay>] [--python </path/to/python>]"
      ),
      call. = FALSE
    )
  }
  parsed
}

graph_to_self_hits <- function(graph_matrix) {
  graph_matrix <- methods::as(graph_matrix, "dgCMatrix")
  graph_summary <- summary(graph_matrix)
  S4Vectors::SelfHits(
    from = as.integer(graph_summary$i),
    to = as.integer(graph_summary$j),
    nnode = ncol(graph_matrix),
    value = as.numeric(graph_summary$x)
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg) == 1L) sub("^--file=", "", script_arg) else NULL
python_root <- normalizePath(file.path(dirname(args$python), ".."), winslash = "/", mustWork = FALSE)
python_lib <- normalizePath(file.path(python_root, "lib"), winslash = "/", mustWork = FALSE)
current_ld_path <- Sys.getenv("LD_LIBRARY_PATH", "")
needs_reexec <- nzchar(python_lib) &&
  file.exists(python_lib) &&
  !startsWith(paste0(current_ld_path, ":"), paste0(python_lib, ":")) &&
  Sys.getenv("SCICEPY_QS_REEXEC", "0") != "1"

if (needs_reexec && !is.null(script_path)) {
  new_ld_path <- if (nzchar(current_ld_path)) {
    paste(python_lib, current_ld_path, sep = ":")
  } else {
    python_lib
  }
  status <- system2(
    command = file.path(R.home("bin"), "Rscript"),
    args = c(script_path, commandArgs(trailingOnly = TRUE)),
    env = c(
      sprintf("LD_LIBRARY_PATH=%s", new_ld_path),
      sprintf("RETICULATE_PYTHON=%s", args$python),
      "SCICEPY_QS_REEXEC=1"
    )
  )
  quit(status = status, save = "no")
}

Sys.setenv(RETICULATE_PYTHON = args$python)

suppressPackageStartupMessages({
  library(Matrix)
  library(qs)
  library(reticulate)
  library(Seurat)
  library(SeuratObject)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(S4Vectors)
  library(zellkonverter)
})

reticulate::use_python(args$python, required = TRUE)
reticulate::import("anndata")

message("Reading Seurat qs: ", args$input_qs)
seurat_obj <- qs::qread(args$input_qs)
if (!inherits(seurat_obj, "Seurat")) {
  stop("Input .qs does not contain a Seurat object.", call. = FALSE)
}

assay_name <- if (!is.null(args$assay)) args$assay else SeuratObject::DefaultAssay(seurat_obj)
if (!assay_name %in% names(seurat_obj@assays)) {
  stop(
    sprintf(
      "Assay '%s' not found. Available assays: %s",
      assay_name,
      paste(names(seurat_obj@assays), collapse = ", ")
    ),
    call. = FALSE
  )
}

graph_name <- if (!is.null(args$graph_name)) args$graph_name else paste0(assay_name, "_snn")
available_graphs <- names(seurat_obj@graphs)
if (!graph_name %in% available_graphs) {
  stop(
    sprintf(
      "Graph '%s' not found. Available graphs: %s",
      graph_name,
      paste(available_graphs, collapse = ", ")
    ),
    call. = FALSE
  )
}

message("Converting Seurat object to SingleCellExperiment using assay: ", assay_name)
sce <- as.SingleCellExperiment(seurat_obj, assay = assay_name)
assay_names <- SummarizedExperiment::assayNames(sce)
if (length(assay_names) == 0L) {
  stop("Converted SingleCellExperiment has no assays.", call. = FALSE)
}
if ("logcounts" %in% assay_names) {
  x_name <- "logcounts"
} else if ("data" %in% assay_names) {
  x_name <- "data"
} else {
  x_name <- assay_names[[1L]]
}

message("Embedding Seurat graphs into colPairs")
for (graph_key in available_graphs) {
  SingleCellExperiment::colPair(sce, graph_key) <- graph_to_self_hits(seurat_obj@graphs[[graph_key]])
}
if (graph_name != "connectivities") {
  SingleCellExperiment::colPair(sce, "connectivities") <- graph_to_self_hits(seurat_obj@graphs[[graph_name]])
}

metadata(sce)$scICEpy_source <- list(
  source_format = "Seurat_qs",
  input_qs = normalizePath(args$input_qs, winslash = "/", mustWork = FALSE),
  assay = assay_name,
  graph_name = graph_name,
  available_graphs = available_graphs,
  x_name = x_name,
  python = args$python
)

output_dir <- dirname(args$output_h5ad)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

message("Writing H5AD: ", args$output_h5ad)
getFromNamespace(".H5ADwriter", "zellkonverter")(
  sce,
  file = args$output_h5ad,
  X_name = x_name,
  skip_assays = FALSE,
  compression = "gzip"
)
message("Done. X assay: ", x_name, "; selected graph alias: connectivities -> ", graph_name)
