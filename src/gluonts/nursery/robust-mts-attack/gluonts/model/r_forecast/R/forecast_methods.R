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

handleQuantileForecast <- function(forecasts, params) {
    outputs = list()
    output_types = params$output_types
    if ("samples" %in% output_types) {
        print("Generating samples are not supported by ``forecast'' package for this forecasting method!
        Use quantiles as output type and pass prediction intervals in the parameters.")
    }
    if("quantiles" %in% output_types) {
        f_upper_matrix <- forecasts$upper
        f_lower_matrix <- forecasts$lower
        outputs$upper_quantiles  <- split(f_upper_matrix, col(f_upper_matrix))
        outputs$lower_quantiles  <- split(f_lower_matrix, col(f_lower_matrix))
    }
    if("mean" %in% output_types) {
        outputs$mean <- forecasts$mean
    }
    outputs
}

handlePointForecast <- function(forecasts, params) {
    outputs = list()
    output_types = params$output_types
    if ("samples" %in% output_types) {
        print("This forecasting method only produces point forecasts! Use mean as (only) output type.")
    }
    if("quantiles" %in% output_types) {
        print("This forecasting method only produces point forecasts! Use mean as (only) output type.")
    }
    if("mean" %in% output_types) {
        outputs$mean <- forecasts$mean
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
    forecasts <- forecast::croston(ts, h=params$prediction_length)
    handlePointForecast(forecasts, params)
}

tbats <- function(ts, params) {
    model <- forecast::tbats(ts)

    # R doesn't allow `simulate` on tbats model. We obtain prediction intervals directly.
    forecasts <- forecast::forecast(model, h=params$prediction_length, level=unlist(params$intervals))
    handleQuantileForecast(forecasts, params)
}

mlp <- function(ts, params) {
    model <- nnfor::mlp(ts, hd.auto.type="valid")

    # `mlp` is a point forecast method.
    forecasts <- forecast::forecast(model, h=params$prediction_length)
    handleForecast(forecasts, params)
}

thetaf <- function(ts, params) {
    # For thetaf, we obtain prediction intervals directly.
    forecasts <- forecast::thetaf(y=ts, h=params$prediction_length, level=unlist(params$intervals))
    handleQuantileForecast(forecasts, params)
}

# Adapted the implementation of STL-AR by Thiyanga Talagala to obtain desired prediction intervals.
# Original implementation: https://rdrr.io/github/thiyangt/seer/src/R/stlar.R
seer_stlar <- function(y, h=10, s.window=11, robust=FALSE, level=c(80, 95))
{
    if(frequency(y)==1 | length(y) <= 2*frequency(y))
        return(forecast::forecast(forecast::auto.arima(y, max.q=0), h=h, level=level))

    fit_stlm <- forecast::stlm(y, s.window=s.window, robust=robust, modelfunction=ar)
    forecast::forecast(fit_stlm, h=h, level=level)
}

stlar <- function(ts, params) {
    h = params$prediction_length
    level = unlist(params$intervals)

    if("s_window" %in% params) {
        s_window = params$s_window
    } else {
        s_window = 11
    }

    if("robust" %in% params) {
        roubst = params$robust
    } else {
        robust = FALSE
    }

    forecasts <- seer_stlar(y=ts, h=h, s.window=s_window, robust=robust, level=level)
    handleQuantileForecast(forecasts, params)
}
