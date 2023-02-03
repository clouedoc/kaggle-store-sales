.PHONY: submit

MESSAGE ?= $(shell bash -c 'read -p "Message: " message; echo $$message')

submit:
	kaggle competitions submit -c store-sales-time-series-forecasting -f out.csv -m "$(MESSAGE)"
