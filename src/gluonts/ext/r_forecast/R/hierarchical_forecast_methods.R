library("hts")
library("parallel")

naive_bottom_up <- function(hts, params) {
    h <- params$prediction_length
    fmethod <- params$fmethod
    fcasts1.bu <- forecast(
      hts,
      h = h,
      method="bu",
      fmethod = fmethod,
      parallel = TRUE,
    )
    aggts(fcasts1.bu)
}

top_down_w_average_historical_proportions <- function(hts, params) {
    h <- params$prediction_length
    fmethod <- params$fmethod
    fcasts1.td <- forecast(
      hts,
      h = h,
      method="tdgsa",
      fmethod = fmethod,
      parallel = TRUE,
    )
    aggts(fcasts1.td)
}

top_down_w_proportions_of_the_historical_averages <- function(hts, params) {
    h <- params$prediction_length
    fmethod <- params$fmethod
    fcasts1.td <- forecast(
      hts,
      h = h,
      method="tdgsf",
      fmethod = fmethod,
      parallel = TRUE,
    )
    aggts(fcasts1.td)
}

top_down_w_forecasts_proportions <- function(hts, params) {
    h <- params$prediction_length
    fmethod <- params$fmethod
    fcasts1.td <- forecast(
      hts,
      h = h,
      method="tdfp",
      fmethod = fmethod,
      parallel = TRUE,
    )
    aggts(fcasts1.td)
}

middle_out_w_forecasts_proportions <- function(hts, params) {
    h <- params$prediction_length
    fmethod <- params$fmethod
    level <- params$level
    fcasts1.mo <- forecast(
      hts,
      h = h,
      method = "mo",
      fmethod = fmethod,
      parallel = TRUE,
      level = level
    )
    aggts(fcasts1.mo)
}

mint <- function(hts, params) {
    h <- params$prediction_length
    fmethod <- params$fmethod
    covariance <- params$covariance
    nonnegative <- params$nonnegative
    algorithm <- params$algorithm

    print(c("covariance:", covariance))

    ally <- aggts(hts)
    n <- nrow(ally)
    p <- ncol(ally)

    allf <- matrix(NA, nrow = h, ncol = p)

    if (tolower(covariance) != "ols") {
        res <- matrix(NA, nrow = n, ncol = p)
    }

    for(i in 1:p)
    {
      fit <- if (fmethod == "arima") auto.arima(ally[, i]) else ets(ally[, i])
      allf[, i] <- forecast(fit, h = h)$mean
      if (tolower(covariance) != "ols") {
          res[, i] <- na.omit(ally[, i] - fitted(fit))
      }
    }

    if (tolower(covariance) == "ols") {
        # OLS is a special case of MinT where the covariance matrix is identity.
        if (is.hts(hts)){
          y.f_cg <- combinef(allf, get_nodes(hts), weights = NULL,
                             keep = "all", algorithms = algorithm, nonnegative = nonnegative, parallel = TRUE)
        } else{
          y.f_cg <- combinef(allf, groups = get_groups(hts), weights = NULL,
                             keep = "all", algorithms = algorithm, nonnegative = nonnegative, parallel = TRUE)
        }
        return(y.f_cg)  # mean forecasts
    } else {
        if (is.hts(hts)){
          y.f_cg <- MinT(allf, get_nodes(hts), residual = res, covariance = covariance,
                         keep = "all", algorithms = algorithm, nonnegative = nonnegative, parallel = TRUE)
        } else{
          y.f_cg <- MinT(allf, groups = get_groups(hts), residual = res, covariance = covariance,
                         keep = "all", algorithms = algorithm, nonnegative = nonnegative, parallel = TRUE)
        }
        return(y.f_cg)  # mean forecasts
    }
}


################################
############## ERM #############
################################
erm_matrix <- function(S, Y, Y_hat){
  # computes erm matrix
  # output : P : bottom time series rows x totaltime series

  S <- as.matrix(S)
  Y <- as.matrix(Y)
  Y_hat <- as.matrix(Y_hat)

  n <- dim(Y_hat)[2]

  temp1 <- solve(t(S)%*%S)
  temp2 <- (t(S)%*%t(Y))%*%Y_hat
  temp3 <- solve(t(Y_hat)%*%Y_hat + 1e-3*diag(n))
  P <- (temp1%*%temp2)%*%temp3

  return(P)
}

get_base_forecasts <- function(ally, h, fmethod){

  p <- ncol(ally)
  allf <- matrix(NA, nrow = h, ncol = p)

  for(i in 1:p)
  {
    fit <- if (fmethod == "arima") auto.arima(ally[, i]) else ets(ally[, i])
    allf[, i] <- forecast(fit, h = h)$mean
  }

  return(allf[nrow(allf),])
}

erm_reconcile_forecasts <- function(P, S, Y_hat){

  Y_hat <- t(as.matrix(Y_hat))
  Y_recon <- t((S%*%P)%*%t(Y_hat))

  return(Y_recon)
}

erm_aux <- function(h, T1, T, p, ally, fmethod, S){

  Y <- ally[seq(T1+h,T),]
  Yhat_h <- matrix(NA, nrow = T-T1-h+1, ncol = p)

  c <- 1 #counter
  for(i in seq(T1,T-h)){
    Yin        <- ally[seq(1,i),]
    Yhat_h[c,] <- get_base_forecasts(Yin, h, fmethod)
    c          <- c + 1
  }

  P    <- erm_matrix(S, Y, Yhat_h)

  Yhat <- get_base_forecasts(ally, h, fmethod)
  Yhat_reconciled <- erm_reconcile_forecasts(P, S, Yhat)

  return(Yhat_reconciled)


}

erm <- function(hts, params){
  # inputs:
  # hts: hierarchical time series object
  # params: params$prediction_length : prediction length
  #       : params$fmethod           : method used for base forecasts: "arima" or "ets"
  #       : params$numcores          : number of cores for parallelization

  ally <-  aggts(hts)
  n <- nrow(ally)
  p <- ncol(ally)

  fmethod <- params$fmethod
  H <- params$prediction_length
  T1 <- n - (params$prediction_length+1)

  if (is.null(params$numcores)) {
    numCores <- detectCores()
  }
  else {
    if(params$numcores > detectCores()){
      stop("params$numcores should be smaller than detectCores()")
      }
    numCores <- params$numcores
  }

  S <- smatrix(hts)
  S <- as.matrix(S)
  ally <- as.matrix(ally)

  T <- dim(ally)[1]
  allf <- matrix(NA, nrow = H, ncol = p)

  Yhat_reconciled <- mclapply(seq(1,H), erm_aux, T1=T1, T=T, p=p, ally=ally, fmethod=fmethod, S=S, mc.cores=numCores)

  for(h in seq(1,H)){
    allf[h,] <- Yhat_reconciled[[h]]
  }

  return(allf)
}