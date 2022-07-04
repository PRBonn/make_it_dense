python_api:
	pip3 install -e .

uninstall:
	pip3 -v uninstall -y make_it_dense

clean:
	git clean -xf . -e data

clean_cache:
	rm -rf ./data/cache/
	rm -rf ./data/models/
