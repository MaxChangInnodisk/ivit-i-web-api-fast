# Integrate streaming with iVIT-I ( WebRTC )
iVIT-I support WebRTC and MSE streaming to integration.

# Features
1. The API about streaming.
2. The WebRTC usage of the stream.

# WebRTC
1. Launch iVIT-I and execute at least one AI task.
   <details style="margin-top:0.5em; margin-bottom:0.5em">
      <summary>Copy the task uid in the url <code>{ip}:{port}/inference/{uid}</code></summary>
      <div style="margin-left:1em;margin-top:1em">
            <img src="./figures/get-task-uid.png">
      </div>
   </details>
2. Modify the [`webrtc.html`](./webrtc.html) with the task uid.
   
   <details style="margin-top:0.5em; margin-bottom:0.5em">
      <summary>Modify the value of the input element ( Line 13 )</summary>
      <div style="margin-left:1em;margin-top:1em">
            <img src="./figures/modify-uid.png">
      </div>
   </details>
3. Move to `webrtc` folder and double click the [`webrtc.html`](./webrtc.html)

   ![cover](./figures/webrtc-sample.png)

   