# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

ARG BASE
FROM ${BASE}

# Install Python Package
RUN pip3 install --no-cache-dir \
    python-dateutil==2.8.2 \
    paho-mqtt==1.6.1 \
    fastapi==0.95.1 \
    "uvicorn[standard]" \
    python-multipart

RUN pip3 install \
    filterpy==1.4.5 \
    lap==0.4.0

RUN pip3 install --no-cache-dir \
    aiofiles \
    streaming-form-data 

CMD [ "bash" ]