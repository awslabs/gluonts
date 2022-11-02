library("hts")

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
