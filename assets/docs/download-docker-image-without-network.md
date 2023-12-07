# iVIT Image for none-network user
1. Prepare the `*.tar` file with each microservice and place it in the same folder, e.g. "./v1.3-images".
    ```bash
    v1.3-images/
    ├── ivit-i-intel:v1.3-service.tar
    ├── ivit-i-nginx:1.23.4-bullseye.tar
    ├── ivit-i-website:v130.tar
    ├── mosquitto:2.0.18.tar
    ├── rtsp-server:v0.23.7.tar
    └── rtsptoweb:latest.tar
    ```
2. Run the script to import each image from the tarball file.
    ```bash
    python3 docker_image_handler.py -m load -f ./v1.3-images
    ```
3. Waiting for the process to be finished.
    ![load-docker-images.png](../images/load-docker-images.png)

