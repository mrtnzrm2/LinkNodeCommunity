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
offset <- (1 - root.position) / 100

p <- ggtree(
  tree, layout = "rectangular", root.position = root.position
) +
  geom_tippoint(size = 3) +
  geom_tiplab(
    size = 5,
    angle = 180, offset = offset, hjust = 1
  )  +
  vexpand(0.025, direction = -1) +
  vexpand(0.025, direction = 1) +
  hexpand(0.025, direction = -1) +
  hexpand(0.025, direction = 1) +
  guides(
    x = guide_axis(minor.ticks = TRUE),
  ) +
  scale_x_continuous(minor_breaks = scales::breaks_width(0.05)) +
  theme_tree2(
    axis.text.x = element_text(size = 30, angle = 90),
    axis.ticks.length = unit(10, "pt"),
    axis.minor.ticks.length = rel(0.5),
    axis.ticks.x = element_line(
      linewidth = 1
    ),
    axis.title.x = element_text(size = 35)
  ) +
  # labs(x = "Interareal distance [mm]")
  # labs(x = TeX("$S=1-\\frac{D}{D_{\\max}}$"))
  labs(x = TeX("$S=1-H^{2}$", italic = TRUE))

device <- "svg"
ggsave(
  paste0(args[2], "_v3", ".", device), device = device,
  width = 10, height = 14,  limitsize = FALSE
)