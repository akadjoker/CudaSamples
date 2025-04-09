SERVICE=jetson



build: 
	docker-compose build

up:
	docker-compose up -d

shell:
	docker-compose run --rm $(SERVICE) /bin/bash

down:
	docker-compose down


clean:
	docker-compose down --volumes --remove-orphans
