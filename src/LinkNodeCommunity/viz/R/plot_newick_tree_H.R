#!/usr/bin/env Rscript
library(ggplot2)
library(latex2exp)
library(ggtree)
library(pracma)

insert_minor <- function(major_labs, n_minor) {
  labs <- c(sapply(major_labs, function(x) c(x, rep("", 4))))
  labs[1:(length(labs) - n_minor)]
}

args <- commandArgs(trailingOnly = TRUE)

tree <- read.tree(text = args[1])
root.position <- as.numeric(args[3])
offset <- 0.02

p <- ggtree(
  tree, layout = "rectangular", root.position = root.position
) +
  geom_tippoint(size = 2.5) +
  geom_tiplab(
    size = 7,
    angle = 180, offset = offset, hjust = 0
  )  +
  vexpand(0.025, direction = -1) +
  vexpand(0.025, direction = 1) +
  hexpand(0.025, direction = -1) +
  hexpand(0.05, direction = 1) +
  guides(
    x = guide_axis(minor.ticks = TRUE),
  ) +
  scale_x_continuous(minor_breaks = scales::breaks_width(0.025)) +
  theme_tree2(
    axis.text.x = element_text(size = 30, angle = 90),
    axis.ticks.length = unit(10, "pt"),
    axis.minor.ticks.length = unit(5, "pt"),  # Add minor ticks length
    axis.ticks.x = element_line(linewidth = 1),
    axis.ticks.x.minor = element_line(linewidth = 0.5),
    axis.ticks.length.x = unit(10, "pt"),     # Major ticks length
    axis.ticks.length.x.minor = unit(5, "pt"), # Minor ticks length
    axis.title.x = element_text(size = 35)
  ) +
  # labs(x = "Interareal distance [mm]")
  # labs(x = TeX("$S=1-\\frac{D}{D_{\\max}}$"))
  labs(x = TeX("$S=1-H^{2}$", italic = TRUE))
  # labs(x = "Jaccard Probability")


device <- "svg"
ggsave(
  paste0(args[2], "_v3", ".", device), device = device,
  width = 11, height = 14,  limitsize = FALSE
)