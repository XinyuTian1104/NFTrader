.PHONY: test
test:
	pytest -s -v --rootdir=$(shell pwd) --cache-clear