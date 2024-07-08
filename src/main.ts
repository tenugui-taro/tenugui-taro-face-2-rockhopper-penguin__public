import "./style.css";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import {
  FaceLandmarksDetector,
  MediaPipeFaceMeshTfjsModelConfig,
} from "@tensorflow-models/face-landmarks-detection";
import { Keypoint } from "@tensorflow-models/face-detection";

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
  });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadModel() {
  const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
  const detectorConfig: MediaPipeFaceMeshTfjsModelConfig = {
    runtime: "tfjs",
    refineLandmarks: false,
  };
  return await faceLandmarksDetection.createDetector(model, detectorConfig);
}

function drawEyebrow(keypoints: Keypoint[]) {
  if (keypoints.length >= 2) {
    ctx.fillStyle = "yellow";
    ctx.beginPath();
    ctx.moveTo(keypoints[0].x, keypoints[0].y);
    keypoints.forEach((keypoint) => {
      ctx.lineTo(keypoint.x, keypoint.y);
    });
    ctx.closePath();
    ctx.fill();
  }
}

async function detectFace(detector: FaceLandmarksDetector) {
  const estimationConfig = {
    flipHorizontal: false,
    staticImageMode: false,
  };
  const faces = await detector.estimateFaces(video, estimationConfig);

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const rightEyebrowKeyPoints = [107, 65, 52, 53, 46, 34, 70, 162, 71, 21, 68];
  const leftEyebrowKeyPoints = [
    336, 295, 282, 283, 276, 264, 300, 389, 301, 251, 298,
  ];

  faces.forEach((face) => {
    const rightEyebrowKeypoints = rightEyebrowKeyPoints.map(
      (index) => face.keypoints[index]
    );
    const leftEyebrowKeypoints = leftEyebrowKeyPoints.map(
      (index) => face.keypoints[index]
    );

    // Draw the eyebrows
    drawEyebrow(rightEyebrowKeypoints);
    drawEyebrow(leftEyebrowKeypoints);
  });
}

async function main() {
  await setupCamera();
  const detector = await loadModel();

  video.play();
  setInterval(() => {
    detectFace(detector);
  }, 100);
}

main();
