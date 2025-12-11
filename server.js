// server.js - Node.js WebRTC + TensorFlow Server
require("dotenv").config();
const express = require("express");
const http = require("http");
const WebSocket = require("ws");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const cors = require("cors");

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(cors());
app.use(express.json({ limit: "50mb" }));

const PORT = process.env.PORT || 8080;
const MODEL_PATH = path.join(__dirname, "model");

let model;

// Load TensorFlow model on startup
(async () => {
  try {
    model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`);
    console.log("✅ Model loaded.");
    console.log("Input shape:", model.inputs[0].shape);
  } catch (err) {
    console.error("❌ Failed to load model:", err);
  }
})();

// Handle WebSocket connections
wss.on("connection", (ws) => {
  console.log("✅ Client connected. Total clients:", wss.clients.size);

  ws.on("message", async (data) => {
    try {
      // Parse incoming message
      const message = JSON.parse(data);

      if (message.type === "frame") {
        // Handle base64 frame data
        const base64Data = message.frameData.split(",")[1];
        const buffer = Buffer.from(base64Data, "base64");

        // Run prediction
        const prediction = await predictLeafDisease(buffer);

        // Send prediction back to client - FIXED STRUCTURE
        ws.send(
          JSON.stringify({
            type: "prediction",
            predictions: prediction.predictions,
            topPrediction: prediction.topPrediction,
            confidence: prediction.confidence,
            timestamp: prediction.timestamp,
          })
        );
      }
    } catch (err) {
      console.error("Error processing frame:", err);
      ws.send(
        JSON.stringify({
          type: "error",
          error: err.message,
          predictions: [],
        })
      );
    }
  });

  ws.on("close", () => {
    console.log("❌ Client disconnected. Total clients:", wss.clients.size);
  });

  ws.on("error", (err) => {
    console.error("WebSocket error:", err);
  });
});

// Prediction function
async function predictLeafDisease(buffer) {
  if (!model) {
    throw new Error("Model not loaded yet");
  }

  return tf.tidy(() => {
    try {
      const tensor = tf.node
        .decodeImage(buffer, 3)
        .resizeNearestNeighbor([224, 224])
        .div(255.0)
        .expandDims();

      const prediction = model.predict(tensor);
      const probabilities = prediction.dataSync();

      const classNames = [
        "Oidium Heveae",
        "Healthy",
        "Anthracnose",
        "Leaf Spot",
      ];

      const predictions = Array.from(probabilities).map((prob, index) => ({
        className: classNames[index],
        probability: prob,
      }));

      const topPrediction = predictions.reduce(
        (max, p) => (p.probability > max.probability ? p : max),
        { className: "", probability: 0 }
      );

      return {
        predictions,
        topPrediction,
        confidence: (topPrediction.probability * 100).toFixed(2),
        timestamp: new Date().toISOString(),
      };
    } catch (err) {
      throw new Error("Prediction failed: " + err.message);
    }
  });
}

// REST endpoints for health check
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    modelLoaded: !!model,
    connectedClients: wss.clients.size,
  });
});

server.listen(PORT, () => {
  console.log(`🚀 Server running at ws://localhost:${PORT}`);
});
