import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";
const socket = io("http://localhost:3001");

export function joinRoom(roomId){ socket.emit("join-room", roomId); }
export function onPeerJoined(cb){ socket.on("peer-joined", cb); }
export function onPeerLeft(cb){ socket.on("peer-left", cb); }
export function sendOffer(roomId, offer){ socket.emit("offer", { roomId, offer }); }
export function sendAnswer(roomId, answer){ socket.emit("answer", { roomId, answer }); }
export function sendIceCandidate(roomId, candidate){ socket.emit("ice-candidate", { roomId, candidate }); }
export function onOffer(cb){ socket.on("offer", cb); }
export function onAnswer(cb){ socket.on("answer", cb); }
export function onIceCandidate(cb){ socket.on("ice-candidate", cb); }
export function sendMicStatus(roomId, enabled){ socket.emit("mic-toggle", { roomId, enabled }); }
export function onMicStatusChanged(cb){ socket.on("mic-toggle", cb); }
