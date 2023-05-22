![COVER](./assets/images/iVIT-I-Logo-B.png)

# iVIT-I-WebAPI-Fast
the faster web api for iVIT-I

# Outline
* [Requirements](#requirements)
* [Quick Start](#quick-start)
* [About Configuration](#about-configuration)
* [About Scripts](#about-scripts)
* [Web API Documentation](#web-api-documentation)
* [Reference](#reference)


# Requirements
* [Docker 20.10 + ](https://docs.docker.com/engine/install/ubuntu/)
* [Docker-Compose v2.15.1 ](https://docs.docker.com/compose/install/linux/#install-using-the-repository)
    * you can check via `docker compose version`


# Quick Start
* Download Repository

    *** **NOTICE: Must use `--recurse-submodules` to clone repository or you gonna lose submodules** ***
    ```bash
    git clone --recurse-submodules https://github.com/MaxChangInnodisk/ivit-i-web-api-fast.git && cd ivit-i-web-api-fast
    ```
    
* Run `iVIT-I-Web-Api`
    ```bash
    sudo ./docker/run.sh -q
    ```

# About Configuration
* [Configuration File: `ivit-i.json`](ivit-i.json)
    | KEY | Desc
    | --- | --- 
    | `PLATFORM`    |   Support key is `intel`, `xilinx`.
    | `SERVICE`     |   Support to modify Web Service `PORT`.
    | `ICAP`        |   Support to modify `HOST`, `PORT`, and `DEVICE_NAME` (iCAP register name).
    | `NGINX`       |   Support to modify Nginx `PORT`.


# About Scripts
* Scripts 
    * Build: [`docker/build.sh`](./docker/build.sh)
    * Run: [`docker/run.sh`](./docker/run.sh)
    * Stop: [`docker/stop.sh`](./docker/stop.sh)

* Run **another platform**

    1. Modify the `PLATFORM` key in [`ivit-i.json`](./ivit-i.json)
        ```json
        "PLATFORM": "xilinx",
        ```
    2. Use `-p` argument: 
        ```bash
        sudo ./docker/run.sh -q -p xilinx
        ```
        *** **NOTICE** ***
        * The argument priority is higher than configuration file ( ivit-i.json )
        * It won't change the configuration content.

* Enter docker container with interative mode.
    ```bash
    # Enter with command line mode
    sudo ./docker/run.sh -qc

    # Run fastapi
    python3 main.py
    ```
* Run in background
    ```bash
    # Background mode
    sudo ./docker/run.sh -qb

    # Close with another script 
    sudo ./docker/stop.sh
    ```
* More Options
    ```bash
    Run the iVIT-I environment.

    Syntax: scriptTemplate [-bcpqh]
    options:
    b               Run in background.
    c               Run command line mode.
    p               Select a platform to run ( the priority is higher than ivit-i.json ). support in [ 'intel', 'xilinx' ]
    q               Qucik start.
    h               help.
    ```

# Web API Documentation
*** *Make sure the web API service has already been launched.* ***
* The documentation will be mounted at `<ip>:<nginx_port>/ivit/docs`
* [FastAPI Swagger ( http://127.0.0.1:6632/ivit/docs )](http://127.0.0.1:6632/ivit/docs)
 

# Reference
* [bluenviron/mediamtx](https://github.com/bluenviron/mediamtx)
* [deepch/RTSPtoWeb](https://github.com/deepch/RTSPtoWeb)
* [nginx](https://www.nginx.com/)