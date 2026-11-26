const CLASSES = ["dry", "ewaste", "medical", "sanitary", "wet"];
const META = {
  dry:      { label: "Dry Waste",      emoji: "🟦", color: "#2979FF", info: "Recyclable materials like paper, plastic and cardboard.", impact: "♻️ High Recycling Potential", reco: "Recycle in Dry Waste Bin" },
  ewaste:   { label: "E-Waste",        emoji: "⚡", color: "#FFAB00", info: "Electronic devices and components.", impact: "☣️ Contains Hazardous Components", reco: "Send to E-Waste Collection Center" },
  medical:  { label: "Medical Waste",  emoji: "🏥", color: "#AA00FF", info: "Potentially infectious healthcare waste.", impact: "🏥 Biomedical Handling Required", reco: "Dispose through Biomedical Waste System" },
  sanitary: { label: "Sanitary Waste", emoji: "🟥", color: "#E53935", info: "Personal hygiene products requiring special handling.", impact: "⚠️ Requires Safe Disposal", reco: "Dispose in Sanitary Waste Bin" },
  wet:      { label: "Wet Waste",      emoji: "🟩", color: "#00C853", info: "Organic waste that can be composted.", impact: "🌱 Compost Friendly", reco: "Dispose in Compost/Wet Waste Bin" },
};

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
let session = null;
let pending = null;

function setStatus(msg, kind) {
  statusEl.className = "status " + kind;
  statusEl.innerHTML = msg;
}

async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status);
  const total = +res.headers.get("content-length") || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got);
    got += value.length;
    $("progress").textContent = Math.round((got / total) * 100) + "%";
  }
  return buf.subarray(0, got);
}

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    const bytes = await fetchWithProgress("/model/waste_model.onnx");
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    setStatus("✅ Model ready — upload or capture a waste image to classify it.", "ready");
    if (pending) { const img = pending; pending = null; classify(img); }
  } catch (e) {
    console.error(e);
    setStatus("❌ Could not load the model: " + e.message + ". Please refresh the page.", "error");
  }
}

function preprocess(img) {
  const c = $("canvas");
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, 224, 224);
  const { data } = ctx.getImageData(0, 0, 224, 224);
  const n = 224 * 224;
  const f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = data[i * 4] / 255;
    f[n + i] = data[i * 4 + 1] / 255;
    f[2 * n + i] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", f, [1, 3, 224, 224]);
}

function softmax(arr) {
  const m = Math.max(...arr);
  const e = arr.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

async function classify(img) {
  $("preview").src = img.src;
  if (!session) {
    pending = img;
    setStatus("⏳ Image received — waiting for the model to finish loading… <span id='progress'></span>", "loading");
    return;
  }
  setStatus("🔄 Classifying…", "loading");
  const out = await session.run({ input: preprocess(img) });
  const probs = softmax(Array.from(out.logits.data));
  render(probs);
  setStatus("✅ Classification complete.", "ready");
}

function render(probs) {
  const best = probs.indexOf(Math.max(...probs));
  const key = CLASSES[best];
  const m = META[key];
  const pct = probs[best] * 100;

  $("results").classList.remove("hidden");
  $("predLabel").innerHTML = `${m.emoji} ${key.toUpperCase()}`;
  $("predLabel").style.background = m.color;
  $("conf").textContent = pct.toFixed(2);
  const g = $("gbar");
  g.style.strokeDasharray = `${pct} 100`;
  g.style.stroke = m.color;

  $("bars").innerHTML = CLASSES.map((c, i) => {
    const p = probs[i] * 100;
    return `<div class="bar-row${i === best ? " best" : ""}">
      <span class="bl">${META[c].emoji} ${c}</span>
      <div class="bt"><div class="bf" style="width:${p}%;background:${META[c].color}"></div></div>
      <span class="bv">${p.toFixed(2)}%</span></div>`;
  }).join("");

  let offset = 25;
  const circles = CLASSES.map((c, i) => {
    const p = probs[i] * 100;
    const el = `<circle cx="21" cy="21" r="15.915" fill="none" stroke="${META[c].color}" stroke-width="6"
      stroke-dasharray="${p} ${100 - p}" stroke-dashoffset="${offset}"><title>${c}: ${p.toFixed(2)}%</title></circle>`;
    offset -= p;
    return el;
  }).join("");
  $("donut").innerHTML = `<circle cx="21" cy="21" r="15.915" fill="none" stroke="#eef2ee" stroke-width="6"/>${circles}
    <text x="21" y="22.5" text-anchor="middle" class="dc">${m.emoji}</text>`;
  $("legend").innerHTML = CLASSES.map((c, i) =>
    `<li><i style="background:${META[c].color}"></i>${META[c].label}<b>${(probs[i] * 100).toFixed(1)}%</b></li>`).join("");

  $("info").textContent = m.info;
  $("impact").textContent = m.impact;
  $("reco").textContent = m.reco;
  $("tbody").innerHTML = CLASSES.map((c, i) =>
    `<tr${i === best ? ' class="best"' : ""}><td>${c}</td><td>${(probs[i] * 100).toFixed(2)}</td></tr>`).join("");

  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function loadFromUrl(url) {
  const img = new Image();
  img.onload = () => classify(img);
  img.onerror = () => setStatus("❌ Could not read that image. Please try a JPG or PNG.", "error");
  img.src = url;
}

function handleFile(file) {
  if (!file || !/^image\/(jpeg|png)$/.test(file.type)) {
    setStatus("⚠️ Please choose a JPG or PNG image.", "error");
    return;
  }
  const r = new FileReader();
  r.onload = () => loadFromUrl(r.result);
  r.readAsDataURL(file);
}

// Upload
$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
$("sample").addEventListener("click", () => loadFromUrl("/sample.jpg"));

// Tabs
let stream = null;
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $("tab-upload").classList.toggle("hidden", t.dataset.tab !== "upload");
  $("tab-camera").classList.toggle("hidden", t.dataset.tab !== "camera");
  if (t.dataset.tab !== "camera" && stream) stopCam();
}));

// Camera
function stopCam() {
  stream.getTracks().forEach((tr) => tr.stop());
  stream = null;
  $("snap").disabled = true;
  $("startCam").textContent = "Start camera";
}
$("startCam").addEventListener("click", async () => {
  if (stream) return stopCam();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    $("video").srcObject = stream;
    $("snap").disabled = false;
    $("startCam").textContent = "Stop camera";
  } catch (e) {
    setStatus("⚠️ Camera unavailable: " + e.message, "error");
  }
});
$("snap").addEventListener("click", () => {
  const v = $("video");
  const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  loadFromUrl(c.toDataURL("image/jpeg", 0.92));
});

loadModel();
