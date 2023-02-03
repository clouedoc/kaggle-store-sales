using CSV
using DataFrames
using Plots

transactions = CSV.read("./data/transactions.csv", DataFrame)

p = plot(transactions[!, :date], 0)
for t in groupby(transactions, :store_nbr)
  store_nbr = t[!, :store_nbr][1]
  if store_nbr > 5
    break
  end
  plot!(t[!, :date], t[!, :transactions], label=store_nbr)
end
display(p)

export transactions