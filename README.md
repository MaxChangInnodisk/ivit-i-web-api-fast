![COVER](./assets/images/iVIT-I-Logo-B.png)

# iVIT-I-WebAPI-Fast
the faster web api for `iVIT-I`

# iVIT-I
iVIT (Vision Intelligence Toolkit) is an AI suite software. you can use iVIT-T to train a custom AI model and deploy to iVIT-I, iVIT-I provides a simpler AI framework and integrate with iCAP, iVIT-I is easy to integrate with your own program by using our `Web API (ivit-i-web-api-fast)` or `Python Library (ivit-i-{platform})`. 

| PLATFORM        | REPOSITORY
| ---             | ---
| Intel           | [ivit-i-intel](https://github.com/InnoIPA/ivit-i-intel)
| Xilinx          | [ivit-i-xilinx](https://github.com/InnoIPA/ivit-i-xilinx)
| Hailo           | [ivit-i-hailo](https://github.com/InnoIPA/ivit-i-hailo)
| NVIDIA dGPU     | [ivit-i-nvidia](https://github.com/InnoIPA/ivit-i-nvidia)
| NVIDIA Jetson   | [ivit-i-jetson](https://github.com/InnoIPA/ivit-i-jetson)


# Outline
- [iVIT-I-WebAPI-Fast](#ivit-i-webapi-fast)
- [iVIT-I](#ivit-i)
- [Outline](#outline)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Prepare Repository](#prepare-repository)
  - [Run `service` quickly with platform.](#run-service-quickly-with-platform)
- [Install Service](#install-service)
- [Configuration](#configuration)
- [About Running Scripts](#about-running-scripts)
- [Web API Documentation](#web-api-documentation)
- [Build Web Site for ARM](#build-web-site-for-arm)
- [Troubleshooting](#troubleshooting)
- [Developement](#developement)
  - [Add new platform](#add-new-platform)
- [Reference](#reference)



# Requirements
*** *Notice: `GPU driver` and `nvidia-docker-toolkit` are required for `NVIDIA` or `Jetson` platforms.* ***

* [Docker 20.10 + ](https://docs.docker.com/engine/install/ubuntu/)
  * `Docker Compose` > `v2.15.X`
    * **[ VERIFY ]** Use this command ( `docker compose version` ).
    * **[ INSTALL ]** Install docker compose by following this [document](https://docs.docker.com/compose/install/linux/#install-using-the-repository) if you don't have docker compose.
    
* ( OPT ) [NVIDIA GPU Driver](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#nvidia-drivers)
* ( OPT ) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#step-1-install-nvidia-container-toolkit)


# Quick Start
## Prepare Repository
```bash
git clone -b r1.1 https://github.com/InnoIPA/ivit-i-web-api-fast.git && cd ivit-i-web-api-fast
```

## Run `service` quickly with platform.
```bash
sudo ./docker/run.sh intel -q
```
* We support `intel`, `xilinx`, `hailo`, `nvidia`, `jetson` now.

# Install Service
We also support `systemctl` to launch iVIT-I when booting. you can select `cli` mode if your system doesn't have GUI and the default value is `gui` if you not set up the mode option.

* Install iVIT-I-Intel service into system, it will auto launch at background
  ```bash
  sudo ./docker/install.sh intel
  ```
  * The usage of the installation script
      ```bash
      Usage:  install.sh [PLATFORM] [MODE].
  
          - PLATFORM: support intel, xilinx, nvidia, jetson, hailo.
          - MODE: support cli, gui. default is gui.
      ```
* Start iVIT-I service
  ```bash
  sudo systemctl start ivit-i
  ```
* Stop iVIT-I service
  ```bash
  sudo systemctl stop ivit-i
  ```
* Check the status of the service
  ```bash
  sudo systemctl status ivit-i
  ```
* Uninstall iVIT-I service
  ```bash
  sudo ./docker/uninstall.sh
  ```

# Configuration
You can modify the configuration file ( [`ivit-i.json`](ivit-i.json) ) to change the port number you want, `SERVICE.PORT` for web service, `NGINX.PORT` for nginx agent, etc.
| KEY           | DESC
| ---           | --- 
| `NGINX`       |   Modify `NGINX` port number if the port number is conflict. default is 6632.
| `SERVICE`     |   Modify `iVIT-I service` port number if the port number is conflict. default is 819.
| `ICAP`        |   Modify HOST and PORT for the `iCAP service`.


# About Running Scripts

* Enter docker container with interative mode.
    ```bash
    # Enter with command line mode
    sudo ./docker/run.sh intel -qc

    # Run fastapi
    python3 main.py
    ```
* Run in background
    ```bash
    # Background mode
    sudo ./docker/run.sh intel -qb

    # Close with another script 
    sudo ./docker/stop.sh
    ```
* More Options
    * Entrance Usage
        ```bash
        Not detect platform !!!!
        Usage     : run.sh [PLATFORM] [OPTION]
        Example   : run.sh intel -q
        ```

    * Each Platform Usage
        ```bash
        $ sudo ./docker/run.sh intel -h
        
        Run the iVIT-I environment.

        Syntax: scriptTemplate [-bcpqh]
        options:
        b               Run in background.
        c               Run command line mode.
        q               Qucik start.
        h               help.
        ```

# Web API Documentation
*** *Make sure the web API service has already been launched.* ***
* The documentation will be mounted at `<ip>:<nginx_port>/ivit/docs`
* [FastAPI Swagger ( http://127.0.0.1:6632/ivit/docs )](http://127.0.0.1:6632/ivit/docs)

# Build Web Site for ARM
Only `aarch64` have to rebuild website service, like `xilinx`, `jetson` platform. More detail please visit [iviti-wa](https://github.com/Jordan00000007/iviti-wa)
    
1. Download repository
    ```bash
    git clone -b v1.0.3 https://github.com/Jordan00000007/iviti-wa && cd iviti-wa
    ```
2. Rebuild docker image
    ```bash
    docker-compose -f ./docker-compose-pro.yml build
    ```

# Troubleshooting
1. Keep re-connect to MQTT ( iCAP ).
    * Issue
        
        ![keeping-connect-to-mqtt-server](assets/images/keeping-connect-to-mqtt-server.png)

    * Solution
        Duplicate name on iCAP server, please modify `DEVICE_NAME` in `ivit-i.json`
        ```JSON
        {
            "ICAP": {
                "DEVICE_NAME": "Your Custom Name"
            }
        }
        ```


# Developement

## Add new platform
1. Modify ivit-i.json
   * `PLATFORM` & `FRAMEWORK`
2. Build docker image.
3. Add a run script that should name with `run-{platform}.sh`
4. Add samples: `./samples/{platform}_sample.py`
   * Update the zip file on the AI Model Zoo which must include a configuration file.
   * Add classification sample
   * Add object detection sample
5. Verify classification sample
6. Verify object detection sample


# Reference
* [bluenviron/mediamtx](https://github.com/bluenviron/mediamtx)
* [deepch/RTSPtoWeb](https://github.com/deepch/RTSPtoWeb)
* [nginx](https://www.nginx.com/)
