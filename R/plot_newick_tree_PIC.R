#!/usr/bin/env Rscript
library(ggtree)
library(ggimage)
library(ape)
library(phyext2)
library(phylobase)
library(phyloTop)
library(adephylo)
library(ggplot2)
library(latex2exp)

args <- commandArgs(trailingOnly = TRUE)

tree <- read.tree(text = args[1])
tree_phylo4 <- as(tree, "phylo4")
picture_path <- args[3]

# Images ----
pic_files <- list.files(picture_path)
pic_files <- paste0(picture_path, "/", pic_files)

tip_data <- data.frame(
  image = pic_files[strtoi(tree_phylo4@label) + 1],
  row.names = nodeId(tree_phylo4, "tip")
)

tree_phylo4 <- phylo4d(tree_phylo4, tip.data = tip_data)

p <- ggtree(
  tree_phylo4, layout = "rectangular",
  branch.length = "none"
)

p <- p +
  xlim(NA, maxHeight(tree_phylo4) + 2) +
  geom_tiplab(
    aes(image = image), geom = "image",
    offset = 2, align = 2, size = .02, angle = -90
  )  +
  geom_tiplab(size = 10, angle = 0, hjust = 0.8) +
  geom_nodepoint(size = 2)

ggsave(args[2], device = "png", width = 10, height = 30,  limitsize = FALSE)