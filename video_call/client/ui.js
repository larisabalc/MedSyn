import { joinRoom, onPeerJoined, onPeerLeft, sendMicStatus, onMicStatusChanged } from "./socket.js";
import { initWebRTC, createOffer, toggleMic, toggleCamera, endCall } from "./webrtc.js";

const params = new URLSearchParams(window.location.search);
const roomId = params.get("room");
const role = params.get("role") || "user";

const joinBtn=document.getElementById("joinBtn");
const joinCard=document.getElementById("joinCard");
const callUI=document.getElementById("callUI");
const videoGrid=document.getElementById("videoGrid");
const peerStatus=document.getElementById("peerStatus");
const micBtn=document.getElementById("micBtn");
const camBtn=document.getElementById("camBtn");
const leaveBtn=document.getElementById("leaveBtn");

let localVideoEl=null;
let remoteVideoEl=null;
let localMic=true;

function createVideoEl(label, muted=false){
    const wrapper=document.createElement("div"); wrapper.className="video-wrapper";
    const video=document.createElement("video"); video.autoplay=true; video.playsInline=true; if(muted) video.muted=true; wrapper.appendChild(video);
    const roleLabel=document.createElement("div"); roleLabel.className="role-label"; roleLabel.innerText=label; wrapper.appendChild(roleLabel);
    const micInd=document.createElement("div"); micInd.className="mic-off-indicator"; micInd.innerHTML=`<i class="bi bi-mic-mute-fill me-1"></i> Mic OFF`; wrapper.appendChild(micInd);
    videoGrid.appendChild(wrapper);
    return {video, micInd, wrapper};
}

joinBtn.onclick=async()=>{
    joinCard.style.display="none"; callUI.style.display="block";

    const {video, wrapper} = createVideoEl(`You: ${role}`, true);
    localVideoEl = video;
    localVideoEl.wrapper = wrapper;

    const remoteLabel = role==="doctor"?"Patient":"Doctor";
    const {video: remoteVideo, wrapper: remoteWrapper} = createVideoEl(remoteLabel);
    remoteVideoEl = remoteVideo;
    remoteVideoEl.wrapper = remoteWrapper;

    await initWebRTC(roomId, localVideoEl, remoteVideoEl);
    joinRoom(roomId);

    detectSpeaking(localVideoEl, localVideoEl.wrapper);
};

onPeerJoined(()=>{ createOffer(); });

onPeerLeft(()=>{
    if(remoteVideoEl && remoteVideoEl.srcObject){
        remoteVideoEl.srcObject.getTracks().forEach(t=>t.stop());
        remoteVideoEl.srcObject = null;
    }
    if(peerStatus) peerStatus.style.display = "block";
});

onMicStatusChanged(({enabled})=>{
    const ind = remoteVideoEl.parentElement.querySelector(".mic-off-indicator");
    if(ind) ind.style.display = enabled?"none":"block";
});

micBtn.onclick=()=>{
    localMic = toggleMic();
    micBtn.innerText = localMic?"Mic ON":"Mic OFF";
    const ind = localVideoEl.parentElement.querySelector(".mic-off-indicator");
    if(ind) ind.style.display = localMic?"none":"block";
    sendMicStatus(roomId, localMic);
};

camBtn.onclick=()=>{
    const cam = toggleCamera();
    camBtn.innerText = cam?"Cam ON":"Cam OFF";
};

leaveBtn.onclick=()=>{
    endCall();
    videoGrid.innerHTML="";
    joinCard.style.display="block";
    callUI.style.display="none";
    if(peerStatus) peerStatus.style.display = "block";
};

function detectSpeaking(videoEl, wrapper){
    if(!videoEl.srcObject) return;
    const audioTracks = videoEl.srcObject.getAudioTracks();
    if(audioTracks.length===0) return;

    const context = new (window.AudioContext||window.webkitAudioContext)();
    const source = context.createMediaStreamSource(videoEl.srcObject);
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);

    setInterval(()=>{
        analyser.getByteFrequencyData(data);
        let sum=0;
        for(let i=0;i<data.length;i++) sum+=data[i];
        const avg = sum/data.length/255;
        if(avg>0.02){
            wrapper.classList.add("speaking");
        } else {
            wrapper.classList.remove("speaking");
        }
    },100);
}
