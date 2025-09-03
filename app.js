// app.js - client-side TF.js face mesh based analyzer (MVP)

let model = null;
const upload = document.getElementById('imageUpload');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const overallScoreEl = document.getElementById('overallScore');
const symmetryEl = document.getElementById('symmetryScore');
const proportionEl = document.getElementById('proportionScore');
const skinEl = document.getElementById('skinScore');
const structureEl = document.getElementById('structureScore');
const adviceEl = document.getElementById('advice');
const resultsDiv = document.getElementById('results');

let img = new Image();

async function loadModel() {
  model = await faceLandmarksDetection.load(faceLandmarksDetection.SupportedPackages.mediapipeFacemesh);
  console.log('model loaded');
}
loadModel();

upload.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  img.onload = () => {
    const maxW = 480, maxH = 640;
    let w = img.width, h = img.height;
    const ratio = Math.min(maxW/w, maxH/h, 1);
    canvas.width = Math.round(w*ratio);
    canvas.height = Math.round(h*ratio);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = url;
});

resetBtn.onclick = () => {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  upload.value = '';
  resultsDiv.classList.add('hidden');
};

analyzeBtn.addEventListener('click', async () => {
  if (!img.src) { alert('Upload a selfie first.'); return; }
  resultsDiv.classList.add('hidden');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const input = tf.browser.fromPixels(canvas);
  const predictions = await model.estimateFaces({input});
  input.dispose();

  if (!predictions || predictions.length === 0) {
    alert('No face detected. Use frontal clear photo in good light.');
    return;
  }
  const face = predictions.sort((a,b)=> (b.box ? b.box.width*b.box.height : 0) - (a.box ? a.box.width*a.box.height : 0))[0];

  drawLandmarks(face);

  const symmetryScore = computeSymmetry(canvas, face);
  const proportionScore = computeProportions(face);
  const structureScore = computeStructure(face);
  const skinScore = await computeSkinHealth(canvas, face);

  const overall = Math.round((symmetryScore*0.35 + proportionScore*0.25 + structureScore*0.2 + skinScore*0.2));

  overallScoreEl.innerText = overall;
  symmetryEl.innerText = symmetryScore;
  proportionEl.innerText = proportionScore;
  skinEl.innerText = skinScore;
  structureEl.innerText = structureScore;

  adviceEl.innerHTML = generateAdvice({symmetryScore, proportionScore, structureScore, skinScore});
  resultsDiv.classList.remove('hidden');
});

function drawLandmarks(face){
  const keypoints = face.scaledMesh || face.annotations && flattenAnnotations(face.annotations);
  if(!keypoints) return;
  ctx.fillStyle = 'rgba(0,150,255,0.9)';
  for(let i=0;i<keypoints.length;i++){
    const [x,y] = keypoints[i];
    ctx.beginPath();
    ctx.arc(x, y, 1.2, 0, Math.PI*2);
    ctx.fill();
  }
}

function flattenAnnotations(ann){
  let out = [];
  for(let k in ann){
    out = out.concat(ann[k]);
  }
  return out;
}

function computeSymmetry(canvasEl, face){
  const c = canvasEl;
  const ctxLocal = c.getContext('2d');
  let box = face.box;
  if(!box){
    const pts = face.scaledMesh;
    const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
    const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
    box = {xMin: minX, yMin:minY, width:maxX-minX, height:maxY-minY};
  } else {
    box = {xMin: box.x, yMin: box.y, width: box.width, height: box.height};
  }
  const pad = 0.1;
  const sx = Math.max(0, box.xMin - box.width*pad);
  const sy = Math.max(0, box.yMin - box.height*pad);
  const sw = Math.min(c.width - sx, box.width*(1 + pad*2));
  const sh = Math.min(c.height - sy, box.height*(1 + pad*2));

  const imgData = ctxLocal.getImageData(sx, sy, sw, sh);
  const w = imgData.width, h = imgData.height;
  let totalDiff = 0, totalPixels = 0;
  for(let y=0;y<h;y++){
    for(let x=0;x<Math.floor(w/2);x++){
      const lx = x, rx = w - 1 - x;
      const li = (y*w + lx)*4, ri = (y*w + rx)*4;
      const lval = 0.21*imgData.data[li] + 0.72*imgData.data[li+1] + 0.07*imgData.data[li+2];
      const rval = 0.21*imgData.data[ri] + 0.72*imgData.data[ri+1] + 0.07*imgData.data[ri+2];
      totalDiff += Math.abs(lval - rval);
      totalPixels++;
    }
  }
  const avgDiff = totalDiff / totalPixels;
  const score = Math.max(0, Math.min(100, Math.round(100 - (avgDiff/60)*100)));
  return score;
}

function computeProportions(face){
  const mesh = face.scaledMesh;
  if(!mesh || mesh.length<10) return 50;
  const xs = mesh.map(p=>p[0]), ys = mesh.map(p=>p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const width = maxX - minX;
  const height = maxY - minY;

  const ratio = height / width;
  const target = 1.62;
  const diff = Math.abs(ratio - target)/target;
  let score = Math.max(0, 100 - diff*100);
  const leftEyeIdx = [33, 133, 159, 145];
  const rightEyeIdx = [362, 263, 386, 374];
  function avgPoint(idxs){
    let sx=0, sy=0;
    idxs.forEach(i=>{sx+=mesh[i][0]; sy+=mesh[i][1];});
    return [sx/idxs.length, sy/idxs.length];
  }
  const left = avgPoint(leftEyeIdx), right = avgPoint(rightEyeIdx);
  const eyeDist = Math.hypot(left[0]-right[0], left[1]-right[1]);
  const eyeToWidth = eyeDist / width;
  const eyeTarget = 0.36;
  const eyeDiff = Math.abs(eyeToWidth - eyeTarget)/eyeTarget;
  score = (score*0.7) + Math.max(0, 100 - eyeDiff*100)*0.3;
  return Math.round(Math.max(10, Math.min(100, score)));
}

function computeStructure(face){
  const mesh = face.scaledMesh;
  if(!mesh) return 50;
  const chin = mesh[152] || mesh[Math.floor(mesh.length/2)];
  const jawL = mesh[234] || mesh[0];
  const jawR = mesh[454] || mesh[mesh.length-1];
  const mid = (jawL[0] + jawR[0]) / 2;
  const chinWidth = Math.abs(jawR[0] - jawL[0]);
  const noseTop = mesh[6] || mesh[1];
  const midFaceWidth = Math.abs(mesh[98] ? mesh[98][0] - mesh[328][0] : chinWidth*0.9);
  const ratio = chinWidth / midFaceWidth;
  const target = 0.9;
  const diff = Math.abs(ratio - target) / target;
  let score = Math.max(0, 100 - diff*160);
  return Math.round(Math.max(5, Math.min(100, score)));
}

async function computeSkinHealth(canvasEl, face){
  const ctxLocal = canvasEl.getContext('2d');
  const mesh = face.scaledMesh;
  const xs = mesh.map(p=>p[0]), ys = mesh.map(p=>p[1]);
  const minX = Math.max(0, Math.floor(Math.min(...xs))), maxX = Math.min(canvasEl.width-1, Math.ceil(Math.max(...xs)));
  const minY = Math.max(0, Math.floor(Math.min(...ys))), maxY = Math.min(canvasEl.height-1, Math.ceil(Math.max(...ys)));
  const w = Math.max(40, maxX - minX), h = Math.max(40, maxY - minY);
  const sampleW = Math.min(120, w), sampleH = Math.min(160, h);
  const tempCanvas = document.createElement('canvas'); tempCanvas.width=sampleW; tempCanvas.height=sampleH;
  const tctx = tempCanvas.getContext('2d');
  tctx.drawImage(canvasEl, minX, minY, w, h, 0, 0, sampleW, sampleH);
  const data = tctx.getImageData(0,0,sampleW,sampleH).data;

  let total=0, count=0;
  function grayAt(i){return (0.21*data[i] + 0.72*data[i+1] + 0.07*data[i+2]);}
  for(let y=1;y<sampleH-1;y++){
    for(let x=1;x<sampleW-1;x++){
      const i = (y*sampleW + x)*4;
      const g = Math.abs(grayAt(i) - grayAt(i-4)) + Math.abs(grayAt(i) - grayAt(i+4)) + Math.abs(grayAt(i) - grayAt(i - sampleW*4)) + Math.abs(grayAt(i) - grayAt(i + sampleW*4));
      total += g;
      count++;
    }
  }
  const avgGrad = total/count;
  const sharpScore = Math.max(0, Math.min(100, Math.round((avgGrad/18)*100)));

  let rSum=0,gSum=0,bSum=0;
  for(let i=0;i<data.length;i+=4){ rSum += data[i]; gSum += data[i+1]; bSum += data[i+2]; }
  const pixels = data.length/4;
  const rAvg = rSum/pixels, gAvg = gSum/pixels;
  const redRatio = (rAvg/(gAvg+1));
  const redPenalty = Math.max(0, Math.min(1, (redRatio - 1.04)/0.5));
  const redScore = Math.max(0, 100 - redPenalty*100);

  let mean = 0;
  const gray = [];
  for(let i=0;i<data.length;i+=4){ gray.push(grayAt(i)); mean += gray[gray.length-1]; }
  mean /= gray.length;
  let variance = 0;
  for(let v of gray) variance += (v - mean)*(v-mean);
  variance /= gray.length;
  const textureScore = Math.max(0, Math.min(100, Math.round(100 - (variance/900)*100)));

  const final = Math.round((sharpScore*0.35) + (redScore*0.2) + (textureScore*0.45));
  return final;
}

function generateAdvice({symmetryScore, proportionScore, structureScore, skinScore}){
  const lines = [];
  if (skinScore < 60) {
    lines.push('<b>Skin</b>: Start with a basic routine — gentle cleanser, moisturizer with SPF, and a targeted acne/spot treatment. Consider dermatologist if severe.');
  } else {
    lines.push('<b>Skin</b>: Looks healthy. Maintain routine: cleanse, moisturize, sunscreen. Add exfoliation 1–2x weekly if needed.');
  }
  if (symmetryScore < 60) {
    lines.push('<b>Symmetry</b>: Natural asymmetry is normal. Non-invasive improvements: hairstyle, facial hair trimming, contouring with makeup. For large concerns, consult specialists.');
  } else {
    lines.push('<b>Symmetry</b>: Symmetry is good — use grooming to highlight strengths (jawline, cheekbones).');
  }
  if (structureScore < 55) {
    lines.push('<b>Structure</b>: Consider posture & fat-loss to enhance jawline. Targeted jawline exercises and strength training can help; long-term: weight loss reduces facial fat.');
  } else {
    lines.push('<b>Structure</b>: Strong bone structure — emphasize with hair & beard styles.');
  }
  if (proportionScore < 60) {
    lines.push('<b>Proportions</b>: Subtle changes (haircut, glasses/frame choice) can improve perceived proportions. Work on hairstyle that elongates or broadens face depending on goal.');
  } else {
    lines.push('<b>Proportions</b>: Good proportions. Keep grooming consistent. Consider style upgrades (fits, collars) to match face shape.');
  }
  return '<ol><li>' + lines.join('</li><li>') + '</li></ol>';
}
