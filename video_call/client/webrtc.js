import { sendOffer, sendAnswer, sendIceCandidate, onOffer, onAnswer, onIceCandidate } from "./socket.js";

let peer = null;
let localStream = null;
let remoteStream = null;
let remoteVideoEl = null;
let roomId = null;
let micEnabled = true;
let camEnabled = true;

const rtcConfig = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };

export async function initWebRTC(r, localVideo, remoteVideo){
    roomId = r;
    remoteVideoEl = remoteVideo;

    localStream = await navigator.mediaDevices.getUserMedia({ video:true, audio:true });
    localVideo.srcObject = localStream;
    await localVideo.play();

    peer = new RTCPeerConnection(rtcConfig);
    localStream.getTracks().forEach(t=>peer.addTrack(t, localStream));

    peer.ontrack = e => {
        remoteStream = e.streams[0];
        remoteVideoEl.srcObject = remoteStream;
        remoteVideoEl.play();

        const peerStatus = document.getElementById("peerStatus");
        if(peerStatus) peerStatus.style.display = "none";
    };

    peer.onicecandidate = e => { if(e.candidate) sendIceCandidate(roomId, e.candidate); };

    peer.onconnectionstatechange = () => {
        if(peer.connectionState === "disconnected" || peer.connectionState === "failed" || peer.connectionState === "closed"){
            clearRemoteVideo();
            const peerStatus = document.getElementById("peerStatus");
            if(peerStatus) peerStatus.style.display = "block";
        }
    };

    onOffer(async offer => {
        await peer.setRemoteDescription(offer);
        const answer = await peer.createAnswer();
        await peer.setLocalDescription(answer);
        sendAnswer(roomId, answer);
    });

    onAnswer(async answer => { await peer.setRemoteDescription(answer); });
    onIceCandidate(async candidate => { if(peer) await peer.addIceCandidate(candidate); });
}

export async function createOffer(){
    if(!peer) return;
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    sendOffer(roomId, offer);
}

export function toggleMic(){
    micEnabled = !micEnabled;
    if(localStream) localStream.getAudioTracks().forEach(t=>t.enabled=micEnabled);
    return micEnabled;
}

export function toggleCamera(){
    camEnabled = !camEnabled;
    if(localStream) localStream.getVideoTracks().forEach(t=>t.enabled=camEnabled);
    return camEnabled;
}

export function clearRemoteVideo(){
    if(remoteVideoEl){
        if(remoteVideoEl.srcObject){
            remoteVideoEl.srcObject.getTracks().forEach(t=>t.stop());
        }
        remoteVideoEl.srcObject = null;
    }
}

export function endCall(){
    if(peer){ 
        peer.close(); peer = null; 
    }
    if(localStream){ 
        localStream.getTracks().forEach(t=>t.stop()); 
        localStream = null; 
    }
    clearRemoteVideo();
}
