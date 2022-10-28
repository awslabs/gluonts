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
      method="mo",
      fmethod = fmethod,
      parallel = TRUE,
      level = level
    )
    aggts(fcasts1.mo)
}
