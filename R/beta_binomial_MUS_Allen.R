#!/usr/bin/env Rscript
library(aod)
library(magrittr)
library(VGAM)
library(MASS)
library(ggplot2)

invlink.2 = function (x, type = c("cloglog", "log", "logit", "probit")) 
{
  switch(type, logit = plogis(x), log = exp(x), cloglog = 1 - 
           exp(-exp(x)), probit = probitlink(x, inverse = TRUE)
  )
}

betabin.2 = function (
  formula, random, data = NULL, link = c("logit", "cloglog", "probit"), 
  phi.ini = NULL, warnings = FALSE, na.action = na.omit, fixpar = list(), 
  hessian = TRUE, control = list(maxit = 2000), ...)  {
  {
    CALL <- mf <- match.call(expand.dots = FALSE)
    tr <- function(string) gsub("^[[:space:]]+|[[:space:]]+$",  # nolint
                                "", string)
    link <- match.arg(link)
    if (length(formula) != 3) 
      stop(paste(tr(deparse(formula)), collapse = " "), "is not a valid formula.")
    else if (substring(deparse(formula)[1], 1, 5) != "cbind") 
      stop(paste(tr(deparse(formula)), collapse = ""), " is not a valid formula.\n",  # nolint: line_length_linter.
           "The response must be a matrix of the form cbind(success, failure)")
    if (length(random) == 3) {
      form <- deparse(random)
      warning("The formula for phi (", form, ") contains a response which is ignored.") # nolint: line_length_linter.
      random <- random[-2]
    }
    explain <- as.character(attr(terms(random), "variables"))[-1]
    if (length(explain) > 1) {
      warning("The formula for phi contains several explanatory variables (", 
              paste(explain, collapse = ", "), ").\n", "Only the first one (", 
              explain[1], ") was considered.")
      explain <- explain[1]
    }
    gf3 <- if (length(explain) == 1) 
      paste(as.character(formula[3]), explain, sep = " + ")
    else as.character(formula[3])
    gf <- formula(paste(formula[2], "~", gf3))
    if (missing(data)) 
      data <- environment(gf)
    mb <- match(c("formula", "data", "na.action"), names(mf), 
                0)
    mfb <- mf[c(1, mb)]
    mfb$drop.unused.levels <- TRUE
    mfb[[1]] <- as.name("model.frame")
    names(mfb)[2] <- "formula"
    mfb <- eval(mfb, parent.frame())
    mt <- attr(mfb, "terms")
    modmatrix.b <- if (!is.empty.model(mt)) 
      model.matrix(mt, mfb)
    else matrix(, NROW(Y), 0)
    Y <- model.response(mfb, "numeric")
    weights <- model.weights(mfb)
    if (!is.null(weights) && any(weights < 0)) 
      stop("Negative wts not allowed")
    n <- rowSums(Y)
    y <- Y[, 1]
    if (any(n == 0)) 
      warning("The data set contains at least one line with weight = 0.\n")
    mr <- match(c("random", "data", "na.action"), names(mf), 
                0)
    mr <- mf[c(1, mr)]
    mr$drop.unused.levels <- TRUE
    mr[[1]] <- as.name("model.frame")
    names(mr)[2] <- "formula"
    mr <- eval(mr, parent.frame())
    if (length(explain) == 0) 
      modmatrix.phi <- model.matrix(object = ~1, data = mr)
    else {
      express <- paste("model.matrix(object = ~ -1 + ", explain, 
                       ", data = mr", ", contrasts = list(", explain, " = 'contr.treatment'))", 
                       sep = "")
      if (is.ordered(data[, match(explain, table = names(mr))])) 
        warning(explain, " is an ordered factor.\n", "Treatment contrast was used to build model matrix for phi.")
      modmatrix.phi <- eval(parse(text = express))
    }
    fam <- eval(parse(text = paste("binomial(link =", link, 
                                   ")")))
    
    fm <- glm(formula = formula, family = fam, data = data, 
              na.action = na.action)
    b <- coef(fm)
    
    if (any(is.na(b))) {
      print(nab <- b[is.na(b)])
      stop("Initial values for the fixed effects contain at least one missing value.")
    }
    nb.b <- ncol(modmatrix.b)
    nb.phi <- ncol(modmatrix.phi)
    if (!is.null(phi.ini) && !(phi.ini < 1 & phi.ini > 0)) 
      stop("phi.ini was set to ", phi.ini, ".\nphi.ini should verify 0 < phi.ini < 1")
    else if (is.null(phi.ini)) 
      phi.ini <- rep(0.1, nb.phi)
    param.ini <- c(b, phi.ini)
    if (!is.null(unlist(fixpar))) 
      param.ini[fixpar[[1]]] <- fixpar[[2]]
    minuslogL <- function(param) {
      if (!is.null(unlist(fixpar))) 
        param[fixpar[[1]]] <- fixpar[[2]]
      b <- param[1:nb.b]
      eta <- as.vector(modmatrix.b %*% b)
      
      
      p <- invlink.2(eta, type = link)
      phi <- as.vector(modmatrix.phi %*% param[(nb.b + 1):(nb.b +
                                                             nb.phi)])
      cnd <- phi == 0
      f1 <- dbinom(x = y[cnd], size = n[cnd], prob = p[cnd],
                   log = TRUE)
      n2 <- n[!cnd]
      y2 <- y[!cnd]
      p2 <- p[!cnd]
      phi2 <- phi[!cnd]
      f2 <- lchoose(n2, y2) + lbeta(
        p2 * (1 - phi2)/phi2 +
          y2, (1 - p2) * (1 - phi2)/phi2 + n2 - y2) - lbeta(p2 * 
            (1 - phi2)/phi2, (1 - p2) * (1 - phi2)/phi2)
      fn <- sum(c(f1, f2))
      if (!is.finite(fn))
        fn <- -1e+20
      -fn
    }
    withWarnings <- function(expr) {
      myWarnings <- NULL
      wHandler <- function(w) {
        myWarnings <<- c(myWarnings, list(w))
        invokeRestart("muffleWarning")
      }
      val <- withCallingHandlers(expr, warning = wHandler)
      list(value = val, warnings = myWarnings)
    }
    
    reswarn <- withWarnings(optim(par = param.ini, fn = minuslogL, 
                                  hessian = hessian, control = control, ...))
    res <- reswarn$value
    if (warnings) {
      if (length(reswarn$warnings) > 0) {
        v <- unlist(lapply(reswarn$warnings, as.character))
        tv <- data.frame(message = v, freq = rep(1, length(v)))
        cat("Warnings during likelihood maximisation:\n")
        print(aggregate(tv[, "freq", drop = FALSE], list(warning = tv$message), 
                        sum))
      }
    }
    param <- res$par
    namb <- colnames(modmatrix.b)
    namphi <- paste("phi", colnames(modmatrix.phi), sep = ".")
    nam <- c(namb, namphi)
    names(param) <- nam
    if (!is.null(unlist(fixpar))) 
      param[fixpar[[1]]] <- fixpar[[2]]
    H <- H.singular <- Hr.singular <- NA
    varparam <- matrix(NA)
    is.singular <- function(X) qr(X)$rank < nrow(as.matrix(X))
    if (hessian) {
      H <- res$hessian
      if (is.null(unlist(fixpar))) {
        H.singular <- is.singular(H)
        if (!H.singular) 
          varparam <- qr.solve(H)
        else warning("The hessian matrix was singular.\n")
      }
      else {
        idparam <- 1:(nb.b + nb.phi)
        idestim <- idparam[-fixpar[[1]]]
        Hr <- as.matrix(H[-fixpar[[1]], -fixpar[[1]]])
        H.singular <- is.singular(Hr)
        if (!H.singular) {
          Vr <- solve(Hr)
          dimnames(Vr) <- list(idestim, idestim)
          varparam <- matrix(rep(NA, NROW(H) * NCOL(H)), 
                             ncol = NCOL(H))
          varparam[idestim, idestim] <- Vr
        }
      }
    }
    else varparam <- matrix(NA)
    if (any(!is.na(varparam))) 
      dimnames(varparam) <- list(nam, nam)
    nbpar <- if (is.null(unlist(fixpar))) 
      sum(!is.na(param))
    else sum(!is.na(param[-fixpar[[1]]]))
    logL.max <- sum(dbinom(x = y, size = n, prob = y/n, log = TRUE))
    logL <- -res$value
    dev <- -2 * (logL - logL.max)
    df.residual <- sum(n > 0) - nbpar
    iterations <- res$counts[1]
    code <- res$convergence
    msg <- if (!is.null(res$message)) 
      res$message
    else character(0)
    if (code != 0) 
      warning("\nPossible convergence problem. Optimization process code: ", 
              code, " (see ?optim).\n")
    new(Class = "glimML", CALL = CALL, link = link, method = "BB", 
        data = data, formula = formula, random = random, param = param, 
        varparam = varparam, fixed.param = param[seq(along = namb)], 
        random.param = param[-seq(along = namb)], logL = logL, 
        logL.max = logL.max, dev = dev, df.residual = df.residual, 
        nbpar = nbpar, iterations = iterations, code = code, 
        msg = msg, singular.hessian = as.numeric(H.singular), 
        param.ini = param.ini, na.action = na.action)
  }
  
}

# paper_labels = c("v1", "v2",  "8l", "v4", "teo", "tepd", "mt",
#                  "dp", "8m", "lip", "7a", "stpc")

# args <- commandArgs(trailingOnly = TRUE)
# sln_path = args[1]
# sln_matrix_name = args[2]
# output_name = args[3]

sln_path <- "/Users/jorgemartinezarmas/Documents/Documents - jmrtnza MacBook Pro/Work/Research/LINKPROJECT/CSV/MUS/Allen"
sln_matrix_name <- "sld.csv"
output_name <- "beta_coeffs.csv"


SLN <- read.csv(paste(sln_path, sln_matrix_name, sep="/"), row.names = 1)

source_areas <- rownames(SLN) %>%
  tolower()
target_areas <- colnames(SLN) %>%
  tolower()

ceil_SLN <- 1 / min(SLN[(SLN > 0) & (!is.na(SLN))])
ceil_SLN <- ceiling(ceil_SLN)

nonzero_sln <- (SLN > 0) & (!is.na(SLN))
SLN <- ceil_SLN * SLN

SLN <- round(SLN)

sln_array  <- SLN[nonzero_sln]

data <- data.frame(
  y = sln_array,
  n = ceil_SLN - sln_array
)

L <- length(sln_array)

betai <- rep(0, L)
betaj <- rep(0, L)

Nrow <- nrow(SLN)
Ncol <- ncol(SLN)

k <- 1
for (j in 1:Ncol) {
  for (i in 1:Nrow) {
    if (nonzero_sln[i, j] > 0) {

      betai[k] <- i
      betaj[k] <- j

      k <- k + 1
    }
  }
}

b.data <- data.frame()

k <- 0
for (i in 2:Ncol) {
  b. <- rep(0, L)
  b.[which(betai == i)] <- 1
  b.[which(betaj == i)] <- -1
  if (k == 0)
    b.data <- data.frame(b.)
  else
    b.data <- cbind(b.data, b.)
  k <- k + 1
}

# for (i in seq_len(nrow(b.data))) {
#   print(sum(b.data[i, ]))
# }

b.data.columns <- paste0(rep("b", Ncol - 1), 2:(Ncol))
colnames(b.data) <- b.data.columns

data <- cbind(data, b.data)

formula = paste("cbind(y, n) ~ -1 +", paste(b.data.columns, collapse = "+")) %>%
  as.formula()

model <- betabin.2(
  formula, ~ 1,
  data = data,
  link = "probit",
  method = "Nelder-Mead",
  control = list(maxit = 1e6)
)

total_coefs <- rep(0, Ncol)
# restricted_paper_aligned = paper_labels_aligned[2:Nrow]

total_coefs <- c(0, coef(model))
names(total_coefs) <- target_areas

se.total_coefs <- c(0, sqrt(diag(vcov(model))))
names(se.total_coefs) <- target_areas

# paper_labels_aligned_w = paper_labels_aligned %>%
#   sub("\\.", "/", .)

write.csv(data.frame(
    # areas = paper_labels_aligned_w[order(-total_coefs)],
    areas = target_areas,
    # beta = unname(total_coefs[order(-total_coefs)])
    beta = unname(total_coefs)
  ),
  file=paste(sln_path, output_name, sep = "/")
)

data.hierarchy <- data.frame(
  areas = target_areas[order(-total_coefs)],
  beta = -total_coefs[order(-total_coefs)],
  se.beta = se.total_coefs[order(-total_coefs)],
  # order = c(0, 0, -1,  1,  2, -2, 0.5, 5, 4, 9, 7, -3)
  order = seq_along(total_coefs)
)

p <- ggplot(data = data.hierarchy, aes(x = order, y = beta, label = areas)) +
  geom_pointrange(aes(ymin=beta-se.beta, ymax=beta+se.beta)) +
  geom_label() +
  theme_bw()

print(sln_path)
ggsave(file=paste(sln_path, "sld_hierarchy_.svg", sep = "/"), plot=p, width=15, height=8)
