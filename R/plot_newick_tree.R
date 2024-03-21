#!/usr/bin/env Rscript
library(ggtree)
library(ape)
library(phylobase)
library(adephylo)
library(ggplot2)
library(latex2exp)

find.levels.tree <- function(tree.edges) {
  ancestors <- c()

  for (i in 1:nrow(tree.edges)) {
    a <- as.character(tree.edges[i, 1])
    d <- as.character(tree.edges[i, 2])
    if (a %in% names(ancestors)) {
      ancestors[d] <- ancestors[a] + 1
    } else {
      ancestors[a] <- 0
      ancestors[d] <- 1
    }
  }
  return(ancestors)
}

settree.edges.color.thr <- function(thr, number.of.nodes, node.levels, tree.acestors) {
  edge.color <- rep("L > th", number.of.nodes)

  if (thr > 0) {
    for (i in 1:length(edge.color)) {
      a.str <- as.character(i)
      if (a.str %in% names(node.levels)) {
        if (node.levels[a.str] >= thr) {
          edge.color[i] <- "L <= th"
        }
      }
    }
  }

  return(edge.color)
}

args <- commandArgs(trailingOnly = TRUE)

tree <- read.tree(text = args[1])
tree.phylo4 <- as(tree, "phylo4")

nodes <- strtoi(args[3])
thr <- as.double(args[4])

node.levels <- find.levels.tree(tree$edge)

tree.ancestors <- c(ancestor(tree.phylo4))
number.of.edges <- length(tree.ancestors)
edge.color <- settree.edges.color.thr(thr, number.of.edges, node.levels, tree.ancestors)

rNodeData <- data.frame(
  color = edge.color,
  row.names = nodeId(tree.phylo4, "all")
)
tree.phylo4 <- phylo4d(tree.phylo4, all.data=rNodeData)

# p <- ggtree(tree.phylo4, layout = "daylight", branch.length = "none", aes(color=color))
p <- ggtree(tree.phylo4, layout = "rectangular", aes(color=color))

p <- p +
  # geom_tiplab(size = 5.5, angle=180, hjust=0.8) +
  geom_nodepoint(aes(color=color), size=4) +
  scale_colour_brewer("color", palette="Dark2") +
  # ylim(c(-15, 16)) +
  # xlim(c(-21, 21)) +
  theme(legend.text=element_text(size=20))

ggsave(args[2], device = "png", width = 10, height = 22,  limitsize = FALSE)