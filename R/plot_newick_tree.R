#!/usr/bin/env Rscript
library(ggtree)
args <- commandArgs(trailingOnly = TRUE)

tree <- read.tree(text = args[1])

p <- ggtree(tree, layout = "rectangular") +
  geom_tippoint() +
  geom_tiplab(size = 4)
  # +
  # theme_tree2()

ggsave(args[2], device = "png", width = 9, height = 8)