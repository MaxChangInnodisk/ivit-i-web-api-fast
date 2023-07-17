
# Troubleshooting

1. Keep re-connect to MQTT ( iCAP ).
    * Issue
        
        ![keeping-connect-to-mqtt-server](../images/keeping-connect-to-mqtt-server.png)

    * Solution
        Duplicate name on iCAP server, please modify `DEVICE_NAME` in `ivit-i.json`
        ```JSON
        {
            "ICAP": {
                "DEVICE_NAME": "Your Custom Name"
            }
        }
        ```
