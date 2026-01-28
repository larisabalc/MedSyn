const { Server } = require("socket.io");

const io = new Server(3001, {
    cors: {
    origin: "*",
    methods: ["GET", "POST"]
    }
});

io.on("connection", socket => {

    socket.on("join-room", roomId => {
        socket.join(roomId);
        socket.to(roomId).emit("peer-joined");
    });

    socket.on("offer", ({ roomId, offer }) => {
        socket.to(roomId).emit("offer", offer);
    });

    socket.on("answer", ({ roomId, answer }) => {
        socket.to(roomId).emit("answer", answer);
    });

    socket.on("ice-candidate", ({ roomId, candidate }) => {
        socket.to(roomId).emit("ice-candidate", candidate);
    });

    socket.on("disconnect", () => {
        console.log("User disconnected");
    });

    socket.on("mic-toggle", ({ roomId, enabled }) => { socket.to(roomId).emit("mic-toggle", { enabled }); });
});

console.log("WebRTC signaling server running on port 3001");
