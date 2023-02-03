using CSV
using DataFrames
using Impute
using Plots

oil = CSV.read("./data/oil.csv", DataFrame)

rename!(oil,
  :dcoilwtico => :oil_price
)

plot(oil[!, :date], Impute.interp(oil[!, :oil_price]), label="Interpolated oil price")
plot!(oil[!, :date], oil[!, :oil_price], label="Oil price")

Impute.interp!(oil)

oil[!, :oil_price][1] = oil[!, :oil_price][2]

if nrow(oil) != nrow(oil |> dropmissing)
  throw(error("[OIL] We didn't fill all the missing rows"))
end

export oil