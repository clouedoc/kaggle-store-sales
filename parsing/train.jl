using CSV
using DataFrames
using MLJ

include("./series_common.jl")

function load_timeseries(path::String)::DataFrame
  ts = CSV.read(path, DataFrame)
  add_time_dummy!(ts)
  join_holidays!(ts)
  join_stores!(ts)
  join_oil!(ts)
  return ts
end

"""Returns y, X"""
function process_for_tree(timeseries::DataFrame, should_unpack::Bool)::Tuple{Vector{Float64},DataFrame}
  X = DataFrame()
  y = []

  columns = [
    :time_dummy,
    :onpromotion,
    :sales
  ]

  if should_unpack
    y, X = unpack(timeseries[!, columns], ==(:sales))
  else
    X = timeseries[!, filter(x -> x != :sales, columns)]
  end

  coerce!(X,
    :time_dummy => Continuous,
    :onpromotion => Count
  )

  #encoder = get_one_hot_encoder(X)
  #X = MLJ.transform(encoder, X)

  return (y, X)
end