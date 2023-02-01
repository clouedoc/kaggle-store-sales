using CSV
using DataFrames
using MLJ
using Plots
using Dates

oil = CSV.read("./data/oil.csv", DataFrame)

rename!(oil,
  :dcoilwtico => :oil_price
)

####### There are missing values for oil price; let's interpolate them

######### LEAD ANG LAG VALUES
## Lag
oil[!, :oil_lag_1] = circshift(oil[!, :oil_price], -1)
oil[!, :oil_lag_3] = circshift(oil[!, :oil_price], -3)
oil[!, :oil_lag_7] = circshift(oil[!, :oil_price], -7)
oil[!, :oil_lag_14] = circshift(oil[!, :oil_price], -14)
oil[!, :oil_lag_30] = circshift(oil[!, :oil_price], -30)
oil[!, :oil_lag_45] = circshift(oil[!, :oil_price], -45)
oil[!, :oil_lag_60] = circshift(oil[!, :oil_price], -60)
oil_def_lag_val = 0
oil[!, :oil_lag_1] = coalesce.(oil[!, :oil_lag_1], oil_def_lag_val)
oil[!, :oil_lag_3] = coalesce.(oil[!, :oil_lag_3], oil_def_lag_val)
oil[!, :oil_lag_7] = coalesce.(oil[!, :oil_lag_7], oil_def_lag_val)
oil[!, :oil_lag_14] = coalesce.(oil[!, :oil_lag_14], oil_def_lag_val)
oil[!, :oil_lag_30] = coalesce.(oil[!, :oil_lag_30], oil_def_lag_val)
oil[!, :oil_lag_45] = coalesce.(oil[!, :oil_lag_45], oil_def_lag_val)
oil[!, :oil_lag_60] = coalesce.(oil[!, :oil_lag_60], oil_def_lag_val)

## Lead
oil[!, :oil_lead_1] = circshift(oil[!, :oil_price], 1)
oil[!, :oil_lead_3] = circshift(oil[!, :oil_price], 3)
oil[!, :oil_lead_7] = circshift(oil[!, :oil_price], 7)
oil[!, :oil_lead_14] = circshift(oil[!, :oil_price], 14)
oil[!, :oil_lead_30] = circshift(oil[!, :oil_price], 30)
oil[!, :oil_lead_45] = circshift(oil[!, :oil_price], 45)
oil[!, :oil_lead_60] = circshift(oil[!, :oil_price], 60)
oil_def_lead_val = 0::Float64
oil[!, :oil_lead_1] = coalesce.(oil[!, :oil_lead_1], oil_def_lead_val)
oil[!, :oil_lead_3] = coalesce.(oil[!, :oil_lead_3], oil_def_lead_val)
oil[!, :oil_lead_7] = coalesce.(oil[!, :oil_lead_7], oil_def_lead_val)
oil[!, :oil_lead_14] = coalesce.(oil[!, :oil_lead_14], oil_def_lead_val)
oil[!, :oil_lead_30] = coalesce.(oil[!, :oil_lead_30], oil_def_lead_val)
oil[!, :oil_lead_45] = coalesce.(oil[!, :oil_lead_45], oil_def_lead_val)
oil[!, :oil_lead_60] = coalesce.(oil[!, :oil_lead_60], oil_def_lead_val)

# conversions
oil[!, :oil_lag_1] = convert.(Float64, oil[!, :oil_lag_1])
oil[!, :oil_lag_3] = convert.(Float64, oil[!, :oil_lag_3])
oil[!, :oil_lag_7] = convert.(Float64, oil[!, :oil_lag_7])
oil[!, :oil_lag_14] = convert.(Float64, oil[!, :oil_lag_14])
oil[!, :oil_lag_30] = convert.(Float64, oil[!, :oil_lag_30])
oil[!, :oil_lag_45] = convert.(Float64, oil[!, :oil_lag_45])
oil[!, :oil_lag_60] = convert.(Float64, oil[!, :oil_lag_60])
oil[!, :oil_lead_1] = convert.(Float64, oil[!, :oil_lead_1])
oil[!, :oil_lead_3] = convert.(Float64, oil[!, :oil_lead_3])
oil[!, :oil_lead_7] = convert.(Float64, oil[!, :oil_lead_7])
oil[!, :oil_lead_14] = convert.(Float64, oil[!, :oil_lead_14])
oil[!, :oil_lead_30] = convert.(Float64, oil[!, :oil_lead_30])
oil[!, :oil_lead_45] = convert.(Float64, oil[!, :oil_lead_45])
oil[!, :oil_lead_60] = convert.(Float64, oil[!, :oil_lead_60])


## First try: FillImputer. Not good!! It fills with really weird values
## Second try: EvoTrees.

EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees

###### Convert dates to a type that can be understood
###### by the model
oil[!, :date_ts] = datetime2unix.(
  Dates.DateTime.(oil[!, :date])
)
coerce!(oil, :date_ts => Continuous)

oil_training = copy(oil)
oil_training
oil_training[!, :oil_price] = coalesce.(oil_training[!, :oil_price], 0)
oil_y, oil_X = unpack(
  oil_training,
  ==(:oil_price)
)

select!(oil_X, Not(:date))
select!(oil_X, Not(:oil_lag_14))
select!(oil_X, Not(:oil_lag_30))
select!(oil_X, Not(:oil_lag_45))
select!(oil_X, Not(:oil_lag_60))
select!(oil_X, Not(:oil_lead_14))
select!(oil_X, Not(:oil_lead_30))
select!(oil_X, Not(:oil_lead_45))
select!(oil_X, Not(:oil_lead_60))

oil_y = convert.(Float64, oil_y)
oil_inputer_mach = machine(
  EvoTreeRegressor(),
  oil_X,
  oil_y
) |> fit!

evaluate(
  EvoTreeRegressor(),
  oil_X,
  oil_y
)


oil_predict_input = select(oil, Not(:date))
plot(oil_predict_input[!, :date_ts], oil_predict_input[!, :oil_price])
oil_predict_input = select(oil_predict_input, Not(:oil_price))
oil_predictions = predict(oil_inputer_mach, oil_predict_input)
plot!(oil_predict_input[!, :date_ts], oil_predictions)


plot(transformed_oil[!, :date], transformed_oil[!, :oil_price])