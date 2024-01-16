const statsEle = document.getElementById('status')
const listEle = document.getElementById('stream-list')
const popEle = document.getElementById('alert')
const streamDivEle = document.getElementById('stream-content')

const uidHead = 'UID: '
const rtcHead = 'RTC: '
const mseHead = 'MES: '

function addText(data) {
  var br = document.createElement("br");
  var content = document.createTextNode(data);
  statsEle.appendChild(content);
  statsEle.appendChild(br);
}

function getStreams() {
  const api = "http://127.0.0.1:8083/streams"
  return fetch(api)
    .then((response) => {
      return response.json()
    })
}

function createCheckbox(content, idx) {
  var div = document.createElement("div")
  var checkbox = document.createElement("input")
  var label = document.createElement("label")

  div.className = "checkbox-wrapper-47"
  checkbox.name = "cb"
  checkbox.id = `cb-${idx}`
  checkbox.value = content
  checkbox.type = "checkbox"

  label.textContent = content
  label.setAttribute('for', `cb-${idx}`);

  div.appendChild(checkbox)
  div.appendChild(label)
  return div
}

async function updateStreamList() {
  const data = await getStreams();
  const streams = Object.keys(data["payload"]);
  if (streams.length == 0) alert('There is no stream available')
  for (id in streams) {
    checkbox = createCheckbox(streams[id], id)
    listEle.appendChild(checkbox)
  }
}

function getCheckedItems() {
  const checkboxes = document.querySelectorAll('input[type=checkbox]');
  let checked = [];
  for (let i = 0; i < checkboxes.length; i++) {
    let checkbox = checkboxes[i];
    if (checkbox.checked) {
      checked.push(checkbox.value)
    }
  }
  return checked;
};


function generateGrid(inputNumber) {
  // var inputNumber = document.getElementById('input-number').value;
  var itemCount = Math.ceil(Math.sqrt(inputNumber));
  var totalItems = itemCount * itemCount;
  var container = document.getElementById('grid-container');

  // 清空現有的 grid
  container.innerHTML = '';

  // 設置列數
  var columns = itemCount;

  for (var i = 0; i < totalItems; i++) {
      var item = document.createElement('div');
      item.id = `grid-item-${i}`;
      // item.className = "grid-item"
      item.style.flex = `1 1 ${100 / columns}%`;
      item.style.display = "flex"
      item.style.alignItems = "center"
      item.style.backgroundColor = "black"
      // item.textContent = i + 1;  // 顯示項目編號

      container.appendChild(item);
  }
}

function startMSE(uid, idx) {

  const mseQueue = []
  let mseSourceBuffer
  let mseStreamingStarted = false

  function startPlay(videoEl, url) {
    const mse = new MediaSource()
    videoEl.src = window.URL.createObjectURL(mse)
    mse.addEventListener('sourceopen', function () {
      const ws = new WebSocket(url)
      ws.binaryType = 'arraybuffer'
      ws.onopen = function (event) {
        addText(`Connected to ${uid}`)
      }
      ws.onmessage = function (event) {
        const data = new Uint8Array(event.data)
        if (data[0] === 9) {
          let mimeCodec
          const decodedArr = data.slice(1)
          if (window.TextDecoder) {
            mimeCodec = new TextDecoder('utf-8').decode(decodedArr)
          } else {
            mimeCodec = Utf8ArrayToStr(decodedArr)
          }
          mseSourceBuffer = mse.addSourceBuffer('video/mp4; codecs="' + mimeCodec + '"')
          mseSourceBuffer.mode = 'segments'
          mseSourceBuffer.addEventListener('updateend', pushPacket)
        } else {
          readPacket(videoEl, event.data)
        }
      }
    }, false)
  }

  function pushPacket(videoEl) {
    let packet

    if (!mseSourceBuffer.updating) {
      if (mseQueue.length > 0) {
        packet = mseQueue.shift()
        mseSourceBuffer.appendBuffer(packet)
      } else {
        mseStreamingStarted = false
      }
    }
    if (videoEl.buffered.length > 0) {
      if (typeof document.hidden !== 'undefined' && document.hidden) {
        // no sound, browser paused video without sound in background
        videoEl.currentTime = videoEl.buffered.end((videoEl.buffered.length - 1)) - 0.5
      }
    }
  }

  function readPacket(videoEl, packet) {
    if (!mseStreamingStarted) {
      mseSourceBuffer.appendBuffer(packet)
      mseStreamingStarted = true
      return
    }
    mseQueue.push(packet)
    if (!mseSourceBuffer.updating) {
      pushPacket(videoEl)
    }
  }

  // Video Element
  var video = document.createElement('video');
  // video.setAttribute('id', 'mse-video');
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true; // 注意：屬性是 camelCase，不是 playsinline
  video.controls = true;
  video.style.width = '100%';
  // video.style.height = '100%';
  // video.style.maxHeight = '100em';

  // var videoContainer = document.createElement('div');
  // videoContainer.className = "video-container"
  
  // var videoContainer = document.createElement
  const container = document.getElementById(`grid-item-${idx}`)
  container.appendChild(video)

  // const videoEl = document.querySelector('#mse-video')
  // ws://localhost:8083/stream/demo/channel/0/mse?uuid=demo&channel=0
  const url = `ws://localhost:8083/stream/${uid}/channel/0/mse?uuid=${uid}&channel=0`
  // fix stalled video in safari
  video.addEventListener('pause', () => {
    if (video.currentTime > video.buffered.end(video.buffered.length - 1)) {
      video.currentTime = video.buffered.end(video.buffered.length - 1) - 0.1
      video.play()
    }
  })
  startPlay(video, url)
}

async function start() {

  const checked_uid = getCheckedItems();

  if (checked_uid.length == 0) {
    alert('Please select a stream')
  }

  generateGrid(checked_uid.length);

  for (idx in checked_uid) {
    const uid = checked_uid[idx]
    // console.log(uid);
    startMSE(uid, idx)
  }

}

document.addEventListener('DOMContentLoaded', updateStreamList);
