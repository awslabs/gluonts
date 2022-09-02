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
