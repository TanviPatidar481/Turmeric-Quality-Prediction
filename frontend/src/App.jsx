import { useEffect, useRef, useState } from "react";
import "./App.css";
import { PRETTY_LABELS, RETAKE_TIPS, predictImage } from "./services/api";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/jpg"];
const INVALID_MSG = "Upload a clear JPG/PNG turmeric powder photo.";

function fileError(file) {
  if (!file) return INVALID_MSG;
  if (!ACCEPTED_TYPES.includes(file.type) && !/\.(jpe?g|png)$/i.test(file.name))
    return INVALID_MSG;
  return null;
}

export default function App() {
  const [preview, setPreview] = useState(null); // object URL
  const [blob, setBlob] = useState(null);
  const [result, setResult] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraFailed, setCameraFailed] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileRef = useRef(null);

  // Start/stop webcam when toggled
  useEffect(() => {
    if (!cameraOn) return;
    let alive = true;
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: "environment" } })
      .then((stream) => {
        if (!alive) return stream.getTracks().forEach((t) => t.stop());
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(() => alive && (setCameraFailed(true), setCameraOn(false)));
    return () => {
      alive = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [cameraOn]);

  function reset() {
    setResult(null);
    setWarnings([]);
    setError("");
  }

  function onFileChosen(e) {
    const file = e.target.files?.[0];
    const msg = fileError(file);
    if (msg) {
      setError(msg);
      return;
    }
    reset();
    setPreview((old) => (old?.startsWith("blob:") && URL.revokeObjectURL(old), URL.createObjectURL(file)));
    setBlob(file);
  }

  function capturePhoto() {
    const video = videoRef.current;
    if (!video?.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(
      (b) => {
        if (!b) return setError(INVALID_MSG);
        reset();
        setPreview((old) => (old?.startsWith("blob:") && URL.revokeObjectURL(old), URL.createObjectURL(b)));
        setBlob(b);
        setCameraOn(false);
      },
      "image/jpeg",
      0.92,
    );
  }

  async function onPredict() {
    if (!blob) return setError(INVALID_MSG);
    setLoading(true);
    reset();
    try {
      const data = await predictImage(blob);
      setResult(data);
      setWarnings(data.warnings ?? (data.confidence < 60 ? ["Low confidence."] : []));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const uncertain = result && result.confidence < 60;

  return (
    <main className="demo">
      <header>
        <h1>Turmeric Powder Classifier</h1>
        <p className="sub">
          Lab prototype: <code>market_bought</code> vs <code>home_ground</code> · MobileNetV2 · 95% test
          accuracy
        </p>
      </header>

      <section className="card">
        <div className="actions">
          <button type="button" onClick={() => fileRef.current?.click()}>
            Upload photo
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => (setCameraFailed(false), setCameraOn((v) => !v))}
          >
            {cameraOn ? "Close camera" : "Use camera"}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            onChange={onFileChosen}
            hidden
          />
        </div>

        {cameraFailed && (
          <p className="error">
            Camera unavailable in this browser — please upload a photo instead.
          </p>
        )}

        {cameraOn && (
          <div className="camera">
            <video ref={videoRef} autoPlay playsInline muted />
            <button type="button" className="secondary" onClick={capturePhoto}>
              Capture
            </button>
          </div>
        )}

        {preview && (
          <figure className="preview">
            <img src={preview} alt="Turmeric powder input preview" />
            <figcaption>Input preview</figcaption>
          </figure>
        )}

        <button
          type="button"
          className="predict"
          disabled={!blob || loading}
          onClick={onPredict}
        >
          {loading ? "Classifying…" : "Predict"}
        </button>

        {error && <p className="error">{error}</p>}

        {(warnings.length > 0 || uncertain) && (
          <p className="warning">
            {uncertain ? "Uncertain — retake recommended. " : ""}
            {warnings.join(" ")} {RETAKE_TIPS}
          </p>
        )}

        {result && (
          <div className="result">
            <h2>{PRETTY_LABELS[result.label] ?? result.label}</h2>
            <div className="confbar">
              <div
                className="fill"
                style={{ width: `${Math.min(Math.max(result.confidence, 0), 100)}%` }}
              />
            </div>
            <p>Confidence: {Number(result.confidence).toFixed(1)}%</p>
            {result.probs && (
              <ul className="probs">
                {Object.entries(result.probs).map(([k, v]) => (
                  <li key={k}>
                    <code>{k}</code>: {Number(v).toFixed(1)}%
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </section>

      <footer>
        <p>
          Backend: <code>POST /predict</code> (multipart image) →{" "}
          <code>{"{label, confidence}"}</code>. Invalid files are rejected, never
          classified.
        </p>
      </footer>
    </main>
  );
}
