using CSV
using DataFrames
using Dates

holidays_events = CSV.read("./data/holidays_events.csv", DataFrame)

rename!(holidays_events,
  :type => :holiday_type,
  :locale => :holiday_locale,
  :locale_name => :holiday_locale_name,
  :description => :holiday_description,
  :transferred => :holiday_transferred
)

holidays_events[!, :holiday_stack_count] = fill(0, holidays_events |> nrows)

# Note when there are multiple holidays on the same date
# Since it might be indicative of something BIG happening
for row in eachrow(holidays_events)
  n_holidays = filter(x -> x.date === row.date, holidays_events) |> nrows
  row[:holiday_stack_count] = n_holidays
end

# Remove duplicate events
unique!(holidays_events, :date)


export holidays_events