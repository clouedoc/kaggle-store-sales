using DataFrames
using Dates
using MLJ

include("holidays.jl")
function join_holidays!(X::DataFrame)::DataFrame
  leftjoin!(X, holidays_events, on=:date)
  X[!, :holiday_type] = coalesce.(X[!, :holiday_type], "Not a holiday")
  X[!, :holiday_locale] = coalesce.(X[!, :holiday_locale], "Not a holiday")
  X[!, :holiday_locale_name] = coalesce.(X[!, :holiday_locale_name], "Not a holiday")
  X[!, :holiday_description] = coalesce.(X[!, :holiday_description], "Not a holiday")
  X[!, :holiday_transferred] = coalesce.(X[!, :holiday_transferred], false)
  X[!, :holiday_stack_count] = coalesce.(X[!, :holiday_stack_count], 0)
  return X
end

include("stores.jl")
function join_stores!(X::DataFrame)::DataFrame
  leftjoin!(X, stores, on=:store_nbr)
  disallowmissing!(
    X,
    [
      :store_city,
      :store_state,
      :store_type,
      :store_cluster
    ],
    error=true
  )
  return X
end

include("oil.jl")
function join_oil!(X::DataFrame)::DataFrame
  leftjoin!(X, oil, on=:date)
  return X
end

function add_time_dummy!(X::DataFrame)::DataFrame
  vec = []

  for row in eachrow(X)
    push!(vec,
      row.date - Date("2013-01-01")
      |>
      Dates.days
    )
  end

  X[!, :time_dummy] = convert.(Int64, vec)
  return X
end

function get_one_hot_encoder(X::DataFrame)::Machine{MLJModels.OneHotEncoder,true}
  encoder = machine(
    OneHotEncoder(
      features=[
        :family
        :store_city
        :store_type
        :holiday_type
        :holiday_locale
        :holiday_locale_name
        :holiday_transferred
        :store_cluster
        :store_nbr
      ]
    ),
    X
  ) |> fit!

  return encoder
end