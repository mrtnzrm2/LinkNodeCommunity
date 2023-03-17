#!/usr/bin/env Rscript
library(ggtree)
args <- commandArgs(trailingOnly = TRUE)

tree <- read.tree(text = args[1])

p <- ggtree(tree) +
  geom_tippoint() +
  geom_tiplab() +
  theme_tree2()

ggsave(args[2], device = "png")