require('dotenv').config()
const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const sdk = require("node-appwrite");
const client = new sdk.Client();


client
    .setEndpoint(process.env.EXPO_PUBLIC_APPWRITE_ENDPOINT)
    .setProject(process.env.EXPO_PUBLIC_APPWRITE_PROJECT_ID)
    .setKey(process.env.EXPO_PUBLIC_APPWRITE_API_KEY);

const account = new sdk.Account(client)

const app = express();
app.use(cors());

const PORT = process.env.PORT || 8080;
const MODEL_PATH = path.join(__dirname, "model");
const UPLOAD_DIR = path.join(__dirname, "uploads");

// Ensure upload directory exists
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Multer setup for image uploads
const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 5 * 1024 * 1024 }, 
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith("image/")) {
      cb(null, true);
    } else {
      cb(new Error("Only image files are allowed!"), false);
    }
  },
});

let model;

// Load the model at startup
(async () => {
  try {
    model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`);
    console.log("✅ Model loaded.");
    console.log("Input shape:", model.inputs[0].shape);
    console.log("Output shape:", model.outputs[0].shape);
  } catch (err) {
    console.error("❌ Failed to load model:", err);
  }
})();

// Predict endpoint
app.post("/predict", upload.single("image"), async (req, res) => {
  if (!model) {
    return res.status(503).json({ error: "Model not loaded yet. Please try again later." });
  }

  if (!req.file) {
    return res.status(400).json({ error: "No image uploaded" });
  }

  const imagePath = req.file.path;

  try {
    console.log(`📸 Processing image: ${req.file.originalname || req.file.filename}`);

    const buffer = fs.readFileSync(imagePath);

    const tensor = tf.node.decodeImage(buffer, 3)
      .resizeNearestNeighbor([224, 224])
      .div(255.0)
      .expandDims();

    console.log("📐 Tensor shape:", tensor.shape);

    let prediction;
    try {
      prediction = model.predict(tensor);
    } catch (err) {
      throw new Error("Model prediction failed: " + err.message);
    }

    const probabilities = (await prediction.array())[0];
    console.log("📊 Raw probabilities:", probabilities);

    const classNames = ["Oidium Heveae", "Healthy", "Anthracnose", "Leaf Spot"];
    if (probabilities.length !== classNames.length) {
      throw new Error(`Expected ${classNames.length} output classes but got ${probabilities.length}`);
    }

    const predictions = probabilities.map((prob, index) => ({
      className: classNames[index],
      probability: prob,
    }));

    const topPrediction = predictions.reduce(
      (max, p) => (p.probability > max.probability ? p : max),
      { className: "", probability: 0 }
    );

    if (topPrediction.probability < 0.6) {
      console.warn("⚠️ Low confidence prediction:", topPrediction);
    }

    const response = {
      tfjsVersion: "1.7.4",
      tmVersion: "2.4.10",
      modelName: "tm-my-image-model",
      labels: classNames,
      predictions,
      topPrediction,
      imageSize: 224,
      timeStamp: new Date().toISOString(),
    };

    console.log("✅ Top prediction:", topPrediction);

    // Delete uploaded image
    try {
      await fs.promises.unlink(imagePath);
    } catch (err) {
      console.error("❌ Error deleting uploaded file:", err);
    }

    return res.json(response);
  } catch (err) {
    console.error("❌ Prediction error:", err.message);

    // Cleanup on failure
    if (fs.existsSync(imagePath)) {
      try {
        await fs.promises.unlink(imagePath);
      } catch (err2) {
        console.error("❌ Failed to delete file after error:", err2.message);
      }
    }

    return res.status(500).json({
      error: "Prediction failed",
      details: err.message,
    });
  }
});

app.get("/verification", async (req, res) => {
  try {
    const { userId, secret, expire } = req.query;
    
    // Validate required parameters
    if (!userId || !secret) {
      return res.status(400).json({
        success: false,
        message: "Missing required parameters"
      });
    }

    if (expire && new Date(expire) < new Date()) {
      return res.status(400).json({
        success: false,
        message: "Verification link has expired"
      });
    }

    const result = await account.updateVerification(userId, secret);
    
    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error("Verification error:", error);
    res.status(400).json({
      success: false,
      message: error.message || "Verification failed"
    });
  }
});


app.get("/predict", (req, res) => {
  res.send(`✅ Server is running and listening on port ${PORT}`);
});

app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});