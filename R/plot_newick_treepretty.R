#!/usr/bin/env Rscript
library(ggtree)
library(ape)
library(ggpp)
library(phylobase)
library(adephylo)
library(ggplot2)
library(latex2exp)

# Define radial nudge function
radial_nudge <- function(x, y, radius = 0.2) {
  angle <- atan2(y, x)
  new_x <- x + radius * cos(angle)
  new_y <- y + radius * sin(angle)
  data.frame(x = new_x, y = new_y)
}

adjacency_to_edge_list <- function(adj_matrix, adj_matrix2) {
  # Find indices of non-zero elements (assuming unweighted or weighted edges)
  edges <- which(adj_matrix != 0, arr.ind = TRUE)
  
  # Create a data frame of edges
  edge_list <- data.frame(
    from = rownames(adj_matrix)[edges[, 1]],
    to = colnames(adj_matrix)[edges[, 2]],
    weight = adj_matrix[edges],
    weight2 = adj_matrix2[edges],
    stringsAsFactors = FALSE
  )
  
  return(edge_list)
}

# Example: Customize arrow style
custom_arrow <- arrow(
  length = unit(0.3, "cm"),
  type = "open",   # "open" or "closed"
  ends = "last",   # "first", "last", or "both"
  angle = 30     # Smaller = sharper angle
  # lineend = "round"
)

my_colormap <- function(values) {
  # Define the color ramp function
  ramp <- colorRamp(c("#ff7c00", "#1ac938", "#023eff"))
  
  # Ensure values are between 0 and 1
  values <- pmin(pmax(values, 0), 1)
  
  # Get RGB colors and convert to hex
  rgb_vals <- ramp(values) / 255
  rgb(rgb_vals[, 1], rgb_vals[, 2], rgb_vals[, 3])
}

# Define the function to add edges between tips
add_tip_edges <- function(p, edges, arrows, offset = 0.2) {
  tree_data <- p$data
  tip_data <- subset(tree_data, isTip)
  
  # Merge coordinates for 'from' tips
  edges_coords <- merge(edges, tip_data, by.x = "from", by.y = "label", suffixes = c("_from", "_to"))
  
  # Merge coordinates for 'to' tips
  edges_coords <- merge(edges_coords, tip_data, by.x = "to", by.y = "label", suffixes = c("_from", "_to"))
  
  # Calculate direction vectors
  dx <- edges_coords$x_to - edges_coords$x_from
  dy <- edges_coords$y_to - edges_coords$y_from
  lengths <- sqrt(dx^2 + dy^2)
  
  # Normalize and apply offset
  edges_coords$x_from_adj <- edges_coords$x_from + offset * dx / lengths
  edges_coords$y_from_adj <- edges_coords$y_from + offset * dy / lengths
  edges_coords$x_to_adj <- edges_coords$x_to - offset * dx / lengths
  edges_coords$y_to_adj <- edges_coords$y_to - offset * dy / lengths

  edges_coords$edge_colors <- my_colormap(edges_coords$weight2)

  edges_coords <- edges_coords[(edges_coords$weight2  < 0.75) & (edges_coords$weight2  > 0.25), ]

  # Add the curves to the plot
  p + geom_curve(
    data = edges_coords,
    aes(x = x_from_adj, y = y_from_adj, xend = x_to_adj, yend = y_to_adj),
    color = edges_coords$edge_colors,
    curvature = -0.1,
    arrow = arrows,
    linewidth = 0.4,
    alpha = 0.4
  )
}

args <- commandArgs(trailingOnly = TRUE)

tree <- read.tree(text = args[1])
root.position <- as.numeric(args[3])
print(root.position)
tree.phylo4 <- as(tree, "phylo4")

# A <- read.csv(paste0(args[4], "FLN40d91.csv"), header = FALSE)
# A <- as.matrix(A)
# A <- A[1:40, 1:40]

# B <- read.csv(paste0(args[4], "sln_matrix.csv"), header = TRUE, row.names = 1)
# B <- as.matrix(B)
# B <- B[1:40, 1:40]

# labelsA <- read.csv(paste0(args[4], "labels40.csv"), header = FALSE)
# labelsA <- as.matrix(labelsA)[1:40]

# rownames(A) <- labelsA
# colnames(A) <- labelsA

# edges <- adjacency_to_edge_list(A, B)

rNodeData <- data.frame(
  row.names = nodeId(tree.phylo4, "all")
)

tree.phylo4 <- phylo4d(tree.phylo4, all.data=rNodeData)

# p <- ggtree(tree.phylo4, layout = "daylight", branch.length = "none")
p <- ggtree(tree, layout = "rectangular", root.position = root.position)

# # Create ggtree object with daylight layout
# p <- ggtree(tree, layout = "daylight")

# Extract the tree data
tree_data <- p$data

# Export relevant columns to CSV
# We select node, parent, x, y, branch length, label, isTip
write.csv(
  tree_data[, c("node", "parent", "x", "y", "branch.length", "label", "isTip")],
  file = paste0(args[4], "tree_data_branch_rec.csv"), row.names = FALSE
)

# p <- p +
#   geom_nodepoint(size = 0) +
#   geom_tippoint(size = 3) +
#   hexpand(0.025, direction = -1) +
#   hexpand(0.05, direction = 1)

# # extract and calculate the nudged positions
# tip_data <- p$data[p$data$isTip, ]
# nudge_df <- do.call(
#   rbind,
#   Map(function(x, y) radial_nudge(x, y, radius = 1.2), tip_data$x, tip_data$y)
# )
# tip_data <- cbind(tip_data, nudge_df)

# #rename duplicated columns
# names(tip_data) <- make.unique(names(tip_data))

# tip_data$angle_adjusted <- ifelse(
#   tip_data$angle > 90 & tip_data$angle < 270,
#   tip_data$angle + 180,
#   tip_data$angle
# )

# p <- p + geom_text(
#   aes(x = x.1, y = y.1, label = label, angle = angle_adjusted),
#   data = tip_data, size = 10, vjust = "center", hjust = "center"
# )

# # p <- add_tip_edges(p, edges, arrows = custom_arrow)

# p <- p +  theme(legend.text = element_text(size = 20))

# print(args[2])
# ggsave(args[2], device = "png", width = 22, height = 10,  limitsize = FALSE)