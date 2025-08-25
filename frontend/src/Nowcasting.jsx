import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * Weather Nowcasting UI (single-file React component, JSX version)
 * --------------------------------------------------
 * - Upload radar/satellite sequences
 * - Predict next frames via mock or API
 * - Compare (side, overlay, difference)
 * - Quick metrics (MSE / SSIM-lite)
 */

function clsx(...args) {
  return args.filter(Boolean).join(" ");
}

function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

// Convert File → base64 dataURL
async function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Draw image to canvas
async function drawImageToCanvas(ctx, dataURL, w, h, opts) {
  const img = new Image();
  img.src = dataURL;
  await new Promise((r) => (img.onload = () => r(null)));
  ctx.save();
  if (opts?.opacity != null) ctx.globalAlpha = opts.opacity;
  const iw = img.width;
  const ih = img.height;
  const scale = Math.min(w / iw, h / ih);
  const dw = iw * scale;
  const dh = ih * scale;
  const dx = (w - dw) / 2;
  const dy = (h - dh) / 2;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, dx, dy, dw, dh);
  ctx.restore();
}

function ssimLite(a, b) {
  let muA = 0,
    muB = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    muA += a[i];
    muB += b[i];
  }
  muA /= n;
  muB /= n;
  let num = 0,
    denA = 0,
    denB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - muA;
    const db = b[i] - muB;
    num += da * db;
    denA += da * da;
    denB += db * db;
  }
  const den = Math.sqrt(denA * denB) + 1e-6;
  return num / den;
}

function canvasToGray(ctx, w, h) {
  const img = ctx.getImageData(0, 0, w, h);
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0, j = 0; i < img.data.length; i += 4, j++) {
    const r = img.data[i];
    const g = img.data[i + 1];
    const b = img.data[i + 2];
    gray[j] = (r * 0.299 + g * 0.587 + b * 0.114) | 0;
  }
  return gray;
}

// Mock predictor
async function mockPredict(inputs, horizon) {
  const out = [];
  const base = inputs[inputs.length - 1];
  const tmp = document.createElement("canvas");
  const ctx = tmp.getContext("2d");
  const img = new Image();
  img.src = base;
  await new Promise((r) => (img.onload = () => r(null)));
  const W = img.width;
  const H = img.height;
  tmp.width = W;
  tmp.height = H;
  for (let t = 1; t <= horizon; t++) {
    ctx.clearRect(0, 0, W, H);
    ctx.filter = "blur(0.6px)";
    ctx.drawImage(img, t * 2, t * 1.5, W, H);
    out.push(tmp.toDataURL("image/png"));
  }
  return out;
}

// --- Main Component --------------------------------------------------------
export default function NowcastingUI() {
  const [mode, setMode] = useState("mock");
  const [endpoint, setEndpoint] = useState("http://localhost:8000");
  const [variable, setVariable] = useState("radar_reflectivity");
  const [horizon, setHorizon] = useState(6);
  const [opacity, setOpacity] = useState(0.6);
  const [compare, setCompare] = useState("side");
  const [threshold, setThreshold] = useState(0);

  const [inputs, setInputs] = useState([]);
  const [truth, setTruth] = useState([]);
  const [preds, setPreds] = useState([]);

  const [playing, setPlaying] = useState(false);
  const [tIndex, setTIndex] = useState(0);
  const [status, setStatus] = useState("Drop input frames to begin");

  const leftCanvasRef = useRef(null);
  const rightCanvasRef = useRef(null);

  const CAN_W = 560;
  const CAN_H = 420;

  const leftFrame = useMemo(() => {
    if (inputs.length > 0) return inputs[inputs.length - 1];
    return undefined;
  }, [inputs]);

  const rightFrame = useMemo(() => {
    if (preds.length === 0) return undefined;
    const idx = Math.min(Math.max(0, tIndex), preds.length - 1);
    return preds[idx];
  }, [preds, tIndex]);

  // Playback loop
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const tick = (now) => {
      const dt = now - last;
      if (dt > 400 && playing && preds.length > 0) {
        setTIndex((i) => (i + 1) % preds.length);
        last = now;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, preds.length]);

  // Render left
  useEffect(() => {
    const c = leftCanvasRef.current;
    if (!c || !leftFrame) return;
    const ctx = c.getContext("2d");
    drawImageToCanvas(ctx, leftFrame, c.width, c.height);
  }, [leftFrame]);

  // Render right
  useEffect(() => {
    const c = rightCanvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    (async () => {
      ctx.clearRect(0, 0, c.width, c.height);
      if (compare === "side") {
        if (rightFrame) await drawImageToCanvas(ctx, rightFrame, c.width, c.height);
      } else if (compare === "overlay") {
        if (leftFrame) await drawImageToCanvas(ctx, leftFrame, c.width, c.height);
        if (rightFrame) await drawImageToCanvas(ctx, rightFrame, c.width, c.height, { opacity });
      } else if (compare === "diff") {
        if (!leftFrame || !rightFrame) return;
        const a = document.createElement("canvas");
        const b = document.createElement("canvas");
        a.width = b.width = c.width;
        a.height = b.height = c.height;
        const actx = a.getContext("2d");
        const bctx = b.getContext("2d");
        await drawImageToCanvas(actx, leftFrame, a.width, a.height);
        await drawImageToCanvas(bctx, rightFrame, b.width, b.height);
        const A = actx.getImageData(0, 0, a.width, a.height);
        const B = bctx.getImageData(0, 0, b.width, b.height);
        const out = ctx.createImageData(c.width, c.height);
        for (let i = 0; i < A.data.length; i += 4) {
          const dr = Math.abs(A.data[i] - B.data[i]);
          const dg = Math.abs(A.data[i + 1] - B.data[i + 1]);
          const db = Math.abs(A.data[i + 2] - B.data[i + 2]);
          let v = (dr + dg + db) / 3;
          if (v < threshold) v = 0;
          const r = Math.min(255, v * 1.5);
          const g = Math.min(255, v);
          const bl = 255 - Math.min(255, v);
          out.data[i] = r;
          out.data[i + 1] = g;
          out.data[i + 2] = bl;
          out.data[i + 3] = 255;
        }
        ctx.putImageData(out, 0, 0);
      }
    })();
  }, [rightFrame, leftFrame, compare, opacity, threshold]);

  async function onPickInput(e) {
    const files = Array.from(e.target.files || []).filter((f) => /image\/(png|jpeg)/.test(f.type));
    if (files.length === 0) return;
    setStatus("Loading input frames...");
    const frames = [];
    for (const f of files.sort((a, b) => a.name.localeCompare(b.name))) {
      frames.push(await fileToDataURL(f));
    }
    setInputs(frames);
    setStatus(`${frames.length} input frame(s) loaded`);
  }

  async function onPickTruth(e) {
    const files = Array.from(e.target.files || []).filter((f) => /image\/(png|jpeg)/.test(f.type));
    if (files.length === 0) return;
    setStatus("Loading ground truth frames...");
    const frames = [];
    for (const f of files.sort((a, b) => a.name.localeCompare(b.name))) {
      frames.push(await fileToDataURL(f));
    }
    setTruth(frames);
    setStatus(`${frames.length} truth frame(s) loaded`);
  }

  async function runPrediction() {
    if (inputs.length === 0) {
      setStatus("Please load input frames first");
      return;
    }
    try {
      setStatus(mode === "mock" ? "Running mock prediction..." : "Calling model API...");
      let out = [];
      if (mode === "mock") {
        out = await mockPredict(inputs, horizon);
      } else {
        const res = await fetch(`${endpoint.replace(/\/$/, "")}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frames: inputs, horizon, variable }),
        });
        if (!res.ok) throw new Error(`API ${res.status}`);
        const data = await res.json();
        if (!data.predictions || !Array.isArray(data.predictions)) throw new Error("Bad API response");
        out = data.predictions;
      }
      setPreds(out);
      setTIndex(0);
      setStatus(`Got ${out.length} predicted frame(s)`);
    } catch (err) {
      console.error(err);
      setStatus(`Prediction failed: ${err?.message || err}`);
    }
  }

  function clearAll() {
    setInputs([]);
    setTruth([]);
    setPreds([]);
    setTIndex(0);
    setStatus("Cleared");
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
      <h1 className="text-2xl font-bold mb-4">🌦️ Weather Nowcasting Demo (JSX)</h1>

      <div className="mb-4">
        <input type="file" accept="image/png,image/jpeg" multiple onChange={onPickInput} />
        <input type="file" accept="image/png,image/jpeg" multiple onChange={onPickTruth} />
        <button onClick={runPrediction} className="ml-2 px-3 py-1 bg-indigo-600 rounded">Predict</button>
        <button onClick={clearAll} className="ml-2 px-3 py-1 bg-slate-700 rounded">Reset</button>
      </div>

      <div>Status: {status}</div>

      <div className="grid grid-cols-2 gap-4 mt-4">
        <canvas ref={leftCanvasRef} width={CAN_W} height={CAN_H} className="border rounded" />
        <canvas ref={rightCanvasRef} width={CAN_W} height={CAN_H} className="border rounded" />
      </div>
    </div>
  );
}
