# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import uuid, socket

def get_mac_address():
    macaddr = uuid.UUID(int = uuid.getnode()).hex[-12:]
    return ":".join([macaddr[i:i+2] for i in range(0,11,2)])

def get_address():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:       
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP
