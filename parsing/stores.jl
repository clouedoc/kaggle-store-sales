using CSV
using DataFrames

stores = CSV.read("./data/stores.csv", DataFrame)

rename!(stores,
  :city => :store_city,
  :state => :store_state,
  :type => :store_type,
  :cluster => :store_cluster
)

describe(stores)
schema(stores)

## Checking for duplicate data
stores_nrow = nrows(stores) # 54
stores_nunique = stores[!, :store_nbr] |> unique |> length # 54
if stores_nrow != stores_nunique
  throw(error("[STORES] There are duplicate stores"))
end

## We're good!

export stores