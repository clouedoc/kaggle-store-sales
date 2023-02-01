using CSV
using DataFrames
using MLJ
using Plots
using Dates

include("./models.jl")

holidays_events = CSV.read("./data/holidays_events.csv", DataFrame)
stores = CSV.read("./data/stores.csv", DataFrame)
test = CSV.read("./data/test.csv", DataFrame)
train = CSV.read("./data/train.csv", DataFrame)
transactions = CSV.read("./data/transactions.csv", DataFrame)
oil = CSV.read("./data/oil.csv", DataFrame)

## Metadata processing

rename!(stores,
  :city => :store_city,
  :state => :store_state,
  :type => :store_type,
  :cluster => :store_cluster
)

rename!(holidays_events,
  :type => :holiday_type,
  :locale => :holiday_locale,
  :locale_name => :holiday_locale_name,
  :description => :holiday_description,
  :transferred => :holiday_transferred
)
rename!(oil,
  :dcoilwtico => :oil_price
)
oil = MLJ.transform(
  machine(FillImputer(), oil) |> fit!,
  oil
)

## Constructing complete time series

X = train

X = innerjoin(X, stores, on=:store_nbr)
X = leftjoin(X, oil, on=:date)
X = leftjoin(X, holidays_events, on=:date)

describe(X)

X[!, :holiday_type] = coalesce.(X[!, :holiday_type], "Not a holiday")
X[!, :holiday_locale] = coalesce.(X[!, :holiday_locale], "Not a holiday")
X[!, :holiday_locale_name] = coalesce.(X[!, :holiday_locale_name], "Not a holiday")
X[!, :holiday_description] = coalesce.(X[!, :holiday_description], "Not a holiday")
X[!, :holiday_transferred] = coalesce.(X[!, :holiday_transferred], false)


X = MLJ.transform(
  machine(FillImputer(features=[:oil_price]), X) |> fit!,
  X
)

schema(X)
y, X = unpack(X, ==(:sales))

# ┌─────────────────────┬────────────────┬──────────┐
# │ names               │ scitypes       │ types    │
# ├─────────────────────┼────────────────┼──────────┤
# │ id                  │ Count          │ Int64    │
# │ date                │ ScientificDate │ Date     │
# │ store_nbr           │ Count          │ Int64    │
# │ family              │ Textual        │ String31 │
# │ onpromotion         │ Count          │ Int64    │
# │ store_city          │ Textual        │ String15 │
# │ store_state         │ Textual        │ String31 │
# │ store_type          │ Textual        │ String1  │
# │ store_cluster       │ Count          │ Int64    │
# │ oil_price           │ Continuous     │ Float64  │
# │ holiday_type        │ Textual        │ String15 │
# │ holiday_locale      │ Textual        │ String15 │
# │ holiday_locale_name │ Textual        │ String31 │
# │ holiday_description │ Textual        │ String   │
# │ holiday_transferred │ Count          │ Bool     │
# └─────────────────────┴────────────────┴──────────┘

# Drop and coerce data to make it ready to be fed into a model
select!(X, Not(:holiday_description))


coerce!(X,
  :store_nbr => OrderedFactor,
  :family => OrderedFactor,
  :store_city => OrderedFactor,
  :store_state => OrderedFactor,
  :store_type => OrderedFactor,
  :store_cluster => OrderedFactor,
  :holiday_type => OrderedFactor,
  :holiday_locale => OrderedFactor,
  :holiday_locale_name => OrderedFactor,
  :holiday_transferred => OrderedFactor
)

schema(X)

#### Encode textual values
# ┌─────────────────────┬───────────────────┬────────────────────────────────────┐
# │ names               │ scitypes          │ types                              │
# ├─────────────────────┼───────────────────┼────────────────────────────────────┤
# │ id                  │ Count             │ Int64                              │
# │ date                │ ScientificDate    │ Date                               │
# │ store_nbr           │ OrderedFactor{54} │ CategoricalValue{Int64, UInt32}    │
# │ family              │ OrderedFactor{33} │ CategoricalValue{String31, UInt32} │
# │ onpromotion         │ Count             │ Int64                              │
# │ store_city          │ OrderedFactor{22} │ CategoricalValue{String15, UInt32} │
# │ store_state         │ OrderedFactor{16} │ CategoricalValue{String31, UInt32} │
# │ store_type          │ OrderedFactor{5}  │ CategoricalValue{String1, UInt32}  │
# │ store_cluster       │ OrderedFactor{17} │ CategoricalValue{Int64, UInt32}    │
# │ oil_price           │ Continuous        │ Float64                            │
# │ holiday_type        │ OrderedFactor{5}  │ CategoricalValue{String15, UInt32} │
# │ holiday_locale      │ OrderedFactor{3}  │ CategoricalValue{String15, UInt32} │
# │ holiday_locale_name │ OrderedFactor{24} │ CategoricalValue{String31, UInt32} │
# │ holiday_transferred │ OrderedFactor{2}  │ CategoricalValue{Bool, UInt32}     │
# └─────────────────────┴───────────────────┴────────────────────────────────────┘


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
X = MLJ.transform(encoder, X)

schema(X)
describe(X)

X[!, :date_ts] = datetime2unix.(
  Dates.DateTime.(X[!, :date])
)
coerce!(X, :date_ts => Continuous)
select!(X, Not(:date))

models(matching(X, y))
schema(X)
tree = EvoTreeRegressor()
mach = machine(tree, X, y) |> fit!
evaluate(tree, X, y)


