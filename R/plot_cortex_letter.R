library(igraph)
library(magrittr)
library(latex2exp)

source_network <- function() {
  E <- matrix(c("X", "A", "X", "B", "X", "C", "Y", "A", "Y", "B", "Y", "C"), ncol = 2, byrow=TRUE)
  netA <- graph_from_edgelist(E)
  coordA <- matrix(c(-0.5, 0, -0.75, -0.5, 0, -0.5, 0.75, -0.5,  0.5, 0), ncol = 2, byrow = TRUE)
  print(E(netA))
  png(file="/Users/jmarti53/Documents/Projects/LINKPROJECT/plots/TOY/cortex_letter/source_network.png")
  par(mar=c(0,0,0,0)+.1)
  plot.igraph(
    netA, layout = coordA,
    vertex.size = 30,
    vertex.label.cex = 3,
    edge.color = "black",
    edge.width=5,
    edge.arrow.size=3,
    edge.label = c(TeX("$n_{XA}$"), TeX("$n_{XB}$"), TeX("$n_{XC}$"), TeX("$n_{YA}$"), TeX("$n_{YB}$"), TeX("$n_{YC}$")),
    edge.label.family = "Arial",
    edge.label.cex = 1.6,
    edge.label.x = c(-0.95, -0.52, 0.23, -0.23, 0.52, 0.95),
    edge.label.y = c(0.2, 0.2, 0.1, 0.1, 0.2, 0.2),
    # edge.curved = -0.2,
    vertex.label.family = "Arial"
  ) 
  
  dev.off()
}

target_network <- function() {
  E <- matrix(c("A", "X", "B", "X", "C", "X", "A", "Y", "B", "Y", "C", "Y"), ncol = 2, byrow=TRUE)
  netA <- graph_from_edgelist(E)
  coordA <- matrix(c(-0.75, -0.5, -0.5, 0,  0, -0.5, 0.75, -0.5,  0.5, 0), ncol = 2, byrow = TRUE)
  print(E(netA))
  png(file="/Users/jmarti53/Documents/Projects/LINKPROJECT/plots/TOY/cortex_letter/target_network.png")
  par(mar=c(0,0,0,0)+.1)
  plot.igraph(
    netA, layout = coordA,
    vertex.size = 30,
    vertex.label.cex = 3,
    edge.color = "black",
    edge.width=5,
    edge.arrow.size=3,
    edge.label = c(TeX("$n_{AX}$"), TeX("$n_{BX}$"), TeX("$n_{CX}$"), TeX("$n_{AY}$"), TeX("$n_{BY}$"), TeX("$n_{CY}$")),
    edge.label.family = "Arial",
    edge.label.cex = 1.6,
    edge.label.x = c(-1, -0.25, 0.52, -0.52, 0.25, 1),
    edge.label.y = c(-0.2, 0.1, -0.2, -0.2, 0.1, -0.2),
    edge.curved = 0,
    vertex.label.family = "Arial"
  )

  dev.off()
}

bow_tie <- function() {
  E <- matrix(c("1", "A", "2", "A", "3", "A", "4", "A", "1", "2", "3", "4"), ncol=2, byrow=2)
  netA <- graph_from_edgelist(E, directed=FALSE)
  coordA <- matrix(c(-0.5, 0.5, 0, 0, -0.5, -0.5, 0.5, 0.5, 0.5, -0.5), ncol = 2, byrow = TRUE)
  
  png(file="/Users/jmarti53/Documents/Projects/LINKPROJECT/plots/TOY/cortex_letter/bow_tie_2.png")
  par(mar=c(0,0,0,0)+.1)
  plot.igraph(
    netA, layout = coordA,
    vertex.size = 30,
    vertex.label.cex = 3,
    edge.color = "black",
    vertex.label.family = "Arial",
    # mark.col = NA,
    mark.col = c("#1f77b4b3", "#ff7f0eb3"),
    mark.shape = -0.3,
    mark.groups = list(c("1", "2", "A"), c("3", "4", "A")),
    mark.border = NA,
    edge.curved = 0,
    edge.width = 5
  )
  
  dev.off()
}

ladder <- function() {
  E <- matrix(c("1", "2", "1", "3", "3", "4", "3", "5", "5", "6", "5", "7"), ncol=2, byrow=2)
  netA <- graph_from_edgelist(E, directed=FALSE)
  coordA <- matrix(
    c(-1, 2, -2, 1, 0, 1, -1, 0, 1, 0, 0, -1, 2, -1),
    ncol = 2, byrow = TRUE)
  
  png(file="/Users/jmarti53/Documents/Projects/LINKPROJECT/plots/TOY/cortex_letter/ladder.png")
  par(mar=c(0,0,0,0)+.1)
  plot.igraph(
    netA, layout = coordA,
    vertex.size = 30,
    vertex.label = NA,
    edge.color = "black",
    mark.border = NA,
    edge.curved = 0,
    edge.width = 5
  )
  
  dev.off()
}

# source_network()
# target_network()
# bow_tie()
# ladder()

simple_4_step1 <- function() {
  g  <- make_empty_graph() %>%
    add.vertices(
      8,
      name = c("A+", "B+", "C+", "D+", "A-", "B-", "C-", "D-"),
      color = c(rep("#c44e52", 4), rep("#4c72b0", 4))
    )
  coordA <- matrix(
    c(
      0, 0.5, 0.5, 0.5, 1, 0.5, 1.5, 0.5,
      0, 0, 0.5, 0, 1, 0, 1.5, 0
    ), ncol = 2, byrow = TRUE
  )
  png(
    file="/Users/jmarti53/Documents/Projects/LINKPROJECT/plots/TOY/cortex_letter/4_dir_vex.png"
  )
  par(mar=c(0,0,0,0)+.1)
  plot.igraph(
    g, layout = coordA,
    vertex.size = 30,
    vertex.label.cex = 2.5,
    vertex.label.family = "Arial",
    vertex.label.color = "white"
  )
}

simple_4_step2 <- function() {
  g  <- make_empty_graph() %>%
    add.vertices(
      8,
      name = c("A+", "B+", "C+", "D+", "A-", "B-", "C-", "D-"),
      color  = c(rep("#c44e52", 4), rep("#4c72b0", 4))
    ) %>%
    add.edges(
      c(
        "A+", "B-", "B+", "B-", "C+", "B-", "D+", "B-"
      ), color = "#4c72b0", width = c(1, 5, 1, 1)
    )
  coordA <- matrix(
    c(
      0, 0.1, 0.5, 0.1, 1, 0.1, 1.5, 0.1,
      0, 0, 0.5, 0, 1, 0, 1.5, 0
    ), ncol = 2, byrow = TRUE
  )
  png(
    file="/Users/jmarti53/Documents/Projects/LINKPROJECT/plots/TOY/cortex_letter/blue_dir_edge.png"
  )
  par(mar=c(0,0,0,0)+.1)
  plot.igraph(
    g, layout = coordA,
    vertex.size = 30,
    vertex.label.cex = 2.5,
    vertex.label.family = "Arial",
    vertex.label.color = "white",
    edge.arrow.size = 3
  )
}

simple_4_step3 <- function() {
  g  <- make_empty_graph() %>%
    add.vertices(
      8,
      name = c("A+", "B+", "C+", "D+", "A-", "B-", "C-", "D-"),
      color  = c(rep("#c44e52", 4), rep("#4c72b0", 4))
    ) %>%
    add.edges(
      c(
        "B+", "A-", "B+", "B-", "B+", "C-", "B+", "D-"
      ), color = "#c44e52", width = c(1, 5, 1, 1)
    )
  coordA <- matrix(
    c(
      0, 0.5, 0.5, 0.5, 1, 0.5, 1.5, 0.5,
      0, 0, 0.5, 0, 1, 0, 1.5, 0
    ), ncol = 2, byrow = TRUE
  )
  png(
    file="/Users/jmarti53/Documents/Projects/LINKPROJECT/plots/TOY/cortex_letter/red_dir_edge.png"
  )
  par(mar=c(0,0,0,0)+.1)
  plot.igraph(
    g, layout = coordA,
    vertex.size = 30,
    vertex.label.cex = 2.5,
    vertex.label.family = "Arial",
    vertex.label.color = "white",
    edge.arrow.size = 3
  )
}

simple_4_step4 <- function() {
  g  <- make_empty_graph() %>%
    add.vertices(
      8,
      name = c("A+", "B+", "C+", "D+", "A-", "B-", "C-", "D-"),
      color  = c(rep("#c44e52", 4), rep("#4c72b0", 4))
    ) %>%
    add.edges(
      c(
        "B+", "A-", "B+", "B-", "B+", "C-", "B+", "D-"
      ),
      color = c("#c44e52", "#4c72b0", "#c44e52", "#c44e52"),
      width = c(1, 5, 1, 1)
    )
  coordA <- matrix(
    c(
      0, 0.5, 0.5, 0.5, 1, 0.5, 1.5, 0.5,
      0, 0, 0.5, 0, 1, 0, 1.5, 0
    ), ncol = 2, byrow = TRUE
  )
  png(
    file="/Users/jmarti53/Documents/Projects/LINKPROJECT/plots/TOY/cortex_letter/rb_dir_edge.png"
  )
  par(mar=c(0,0,0,0)+.1)
  plot.igraph(
    g, layout = coordA,
    vertex.size = 30,
    vertex.label.cex = 2.5,
    vertex.label.family = "Arial",
    vertex.label.color = "white",
    edge.arrow.size = 3
  )
}

simple_4_step5 <- function() {
  g  <- make_empty_graph() %>%
    add.vertices(
      8,
      name = c("A+", "B+", "C+", "D+", "A-", "B-", "C-", "D-"),
      color  = c(rep("#c44e52", 4), rep("#4c72b0", 4))
    ) %>%
    add.edges(
      c(
        "A+", "B-", "B+", "B-", "C+", "B-", "D+", "B-"
      ),
      color = c("#4c72b0", "#c44e52", "#4c72b0", "#4c72b0"),
      width = c(1, 5, 1, 1)
    )
  coordA <- matrix(
    c(
      0, 0.5, 0.5, 0.5, 1, 0.5, 1.5, 0.5,
      0, 0, 0.5, 0, 1, 0, 1.5, 0
    ), ncol = 2, byrow = TRUE
  )
  png(
    file="/Users/jmarti53/Documents/Projects/LINKPROJECT/plots/TOY/cortex_letter/br_dir_edge.png"
  )
  par(mar=c(0,0,0,0)+.1)
  plot.igraph(
    g, layout = coordA,
    vertex.size = 30,
    vertex.label.cex = 2.5,
    vertex.label.family = "Arial",
    vertex.label.color = "white",
    edge.arrow.size = 3
  )
}

simple_4_step2()
# simple_4_step3()
# simple_4_step4()
# simple_4_step5()