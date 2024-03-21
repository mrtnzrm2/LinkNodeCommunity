#!/usr/bin/env Rscript
library(ggplot2)
library(latex2exp)
library(ggtree)

args <- commandArgs(trailingOnly = TRUE)

tree <- read.tree(text = args[1])
root.position <- as.numeric(args[3])

p <- ggtree(
  tree, layout = "rectangular", root.position = root.position
) +
  geom_tippoint(size = 4) +
  geom_tiplab(size = 6, angle = 0, offset = (1 - root.position) / 100)  +
  vexpand(0.01, direction = -1) +
  hexpand(0.01, direction = -1) +
  theme_tree2(axis.text.x = element_text(size = 20))

device <- "svg"
ggsave(
  paste0(args[2], ".", device), device = device,
  width = 10, height = 14,  limitsize = FALSE
)