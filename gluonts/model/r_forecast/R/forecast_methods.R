loadNamespace("forecast")

handleForecast <- function(model, params) {
    outputs = list()
    output_types = params$output_types
    if ("samples" %in% output_types) {
        outputs$samples <- lapply(1:params$num_samples, function(n) { simulate(model, params$prediction_length) } )
    }
    if("quantiles" %in% output_types) {
        f_matrix <- forecast::forecast(model, h=params$prediction_length, level=unlist(params$levels))$upper
        outputs$quantiles <- split(f_matrix, col(f_matrix))
    }
    if("mean" %in% output_types) {
        outputs$mean <- forecast::forecast(model, h=params$prediction_length)$mean
    }
    outputs
}


arima <- function(ts, params) {
    model <- forecast::auto.arima(ts, trace=TRUE)
    handleForecast(model, params)
}

ets <- function(ts, params) {
    model <- forecast::ets(ts, additive.only=TRUE)
    handleForecast(model, params)
}

croston <- function(ts, params) {
    model <- forecast::croston(ts)
    handleForecast(model, params)
}

tbats <- function(ts, params) {
    model <- forecast::tbats(ts)
    handleForecast(model, params)
}

mlp <- function(ts, params) {
    model <- nnfor::mlp(ts, hd.auto.type="valid")
    handleForecast(model, params)
}
