IMAGE_UPLOADER_IMAGE_NAME=zeals_streamlit
IMAGE_UPLOADER_CONTAINER_NAME=zeals_streamlit

build:
	docker build -t ${IMAGE_UPLOADER_IMAGE_NAME} --force-rm=true .

run:
	docker run --rm --name ${IMAGE_UPLOADER_CONTAINER_NAME} -v ${PWD}:/workspace -w /workspace -it ${IMAGE_UPLOADER_IMAGE_NAME} bash

exec:
	docker exec -it ${IMAGE_UPLOADER_IMAGE_NAME} bash