/**
 * FaceShield — Server-Side Face Blur (Node.js)
 *
 * Produces the EXACT same result as the client-side index.html.
 *
 * Usage:
 *   node blur-faces.js <input-image> [output-image]
 *
 * Example:
 *   node blur-faces.js photo.jpg blurred.png
 *
 * Dependencies:
 *   npm install @vladmandic/face-api canvas
 */

const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage, ImageData } = require('canvas');
const faceapi = require('@vladmandic/face-api');

// ─── Configuration (same defaults as the client-side code) ───────────────────
const BLUR_RADIUS = 15;       // px — strength of the Gaussian blur
const MIN_CONFIDENCE = 0.4;   // 0–1 — detection score threshold
const FEATHER_RADIUS = 34;    // px — softness of the blur edge

// ─── Patch: tell face-api.js to use node-canvas ─────────────────────────────
// This is the key step — in the browser, face-api uses DOM canvas automatically.
// On Node.js, we need to monkey-patch the environment.
const { Canvas, Image } = require('canvas');
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// ─── Separable Gaussian Blur (identical to client-side) ──────────────────────
// True 2-pass convolution: horizontal then vertical.
// Kernel: e^(−x²/2σ²), σ = radius/2.5, half-width = ceil(3σ).
// Edge pixels use clamp-to-edge padding — no dark borders.
function separableGaussianBlur(src, w, h, radius) {
    const sigma = Math.max(1, radius / 2.5);
    const kHalf = Math.ceil(sigma * 3);
    const kLen = kHalf * 2 + 1;
    const kernel = new Float32Array(kLen);
    let kSum = 0;
    for (let i = 0; i < kLen; i++) {
        const x = i - kHalf;
        kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
        kSum += kernel[i];
    }
    for (let i = 0; i < kLen; i++) kernel[i] /= kSum; // normalise

    const tmp = new Float32Array(src.length);
    const out = new Uint8ClampedArray(src.length);

    // ── Horizontal pass: src → tmp ──────────────────────────────────────────
    for (let row = 0; row < h; row++) {
        for (let col = 0; col < w; col++) {
            let r = 0, g = 0, b = 0, a = 0;
            for (let k = -kHalf; k <= kHalf; k++) {
                const sc = Math.min(w - 1, Math.max(0, col + k));
                const idx = (row * w + sc) * 4;
                const wt = kernel[k + kHalf];
                r += src[idx] * wt;
                g += src[idx + 1] * wt;
                b += src[idx + 2] * wt;
                a += src[idx + 3] * wt;
            }
            const oi = (row * w + col) * 4;
            tmp[oi] = r; tmp[oi + 1] = g; tmp[oi + 2] = b; tmp[oi + 3] = a;
        }
    }

    // ── Vertical pass: tmp → out ────────────────────────────────────────────
    for (let col = 0; col < w; col++) {
        for (let row = 0; row < h; row++) {
            let r = 0, g = 0, b = 0, a = 0;
            for (let k = -kHalf; k <= kHalf; k++) {
                const sr = Math.min(h - 1, Math.max(0, row + k));
                const idx = (sr * w + col) * 4;
                const wt = kernel[k + kHalf];
                r += tmp[idx] * wt;
                g += tmp[idx + 1] * wt;
                b += tmp[idx + 2] * wt;
                a += tmp[idx + 3] * wt;
            }
            const oi = (row * w + col) * 4;
            out[oi] = r; out[oi + 1] = g; out[oi + 2] = b; out[oi + 3] = a;
        }
    }
    return out;
}

// ─── Main ────────────────────────────────────────────────────────────────────
async function blurFaces(inputPath, outputPath) {
    console.log(`\n🛡️  FaceShield — Server-Side Face Blur`);
    console.log(`─────────────────────────────────────────`);
    console.log(`Input:  ${inputPath}`);
    console.log(`Output: ${outputPath}`);
    console.log(`Blur:   ${BLUR_RADIUS}px  |  Feather: ${FEATHER_RADIUS}px  |  Confidence: ${MIN_CONFIDENCE}\n`);

    // ── 1. Load models ────────────────────────────────────────────────────────
    // Use local ./models folder first, fall back to the npm package's bundled models
    const localModelDir = path.resolve(__dirname, 'models');
    let modelPath;

    if (fs.existsSync(localModelDir)) {
        modelPath = `file://${localModelDir}`;
        console.log('📦 Loading models from local ./models folder…');
    } else {
        // Use models bundled with the @vladmandic/face-api npm package
        modelPath = `file://${path.resolve(__dirname, 'node_modules/@vladmandic/face-api/model')}`;
        console.log('📦 Loading models from @vladmandic/face-api package…');
    }

    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath.replace('file://', ''));
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath.replace('file://', ''));
    console.log('✅ Models loaded\n');

    // ── 2. Load the input image ───────────────────────────────────────────────
    const img = await loadImage(inputPath);
    const w = img.width;
    const h = img.height;
    console.log(`🖼️  Image: ${w}×${h}px`);

    // Draw original onto the main canvas
    const canvas = createCanvas(w, h);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // ── 3. Detect faces ───────────────────────────────────────────────────────
    console.log('🔍 Running SSD MobileNet v1 + Landmark68…');
    const detections = await faceapi
        .detectAllFaces(canvas, new faceapi.SsdMobilenetv1Options({ minConfidence: MIN_CONFIDENCE }))
        .withFaceLandmarks();

    console.log(`   Detected ${detections.length} face(s)`);

    if (detections.length === 0) {
        console.log('⚠️  No faces found. Try a clearer photo or lower MIN_CONFIDENCE.');
        // Save unmodified image
        const buf = canvas.toBuffer('image/png');
        fs.writeFileSync(outputPath, buf);
        console.log(`💾 Saved (unmodified): ${outputPath}\n`);
        return;
    }

    // ── 4. Create fully-blurred version (same as client side) ─────────────────
    const blurredCanvas = createCanvas(w, h);
    const blurredCtx = blurredCanvas.getContext('2d');
    blurredCtx.drawImage(img, 0, 0);
    const srcData = blurredCtx.getImageData(0, 0, w, h);
    const gaussData = separableGaussianBlur(srcData.data, w, h, BLUR_RADIUS);
    blurredCtx.putImageData(new ImageData(gaussData, w, h), 0, 0);

    // ── 5. For each face: feathered elliptical blur (identical logic) ─────────
    detections.forEach((det, i) => {
        const box = det.detection.box;

        // Expand bounding box (same padding as client)
        const padX = box.width * 0.2;
        const padY = box.height * 0.2;
        const fx = Math.max(0, box.x - padX);
        const fy = Math.max(0, box.y - padY * 1.5);
        const fw = Math.min(w - fx, box.width + padX * 2);
        const fh = Math.min(h - fy, box.height + padY * 2.5);

        // 1. Mask: opaque ellipse over face region
        const maskCanvas = createCanvas(w, h);
        const maskCtx = maskCanvas.getContext('2d');
        maskCtx.clearRect(0, 0, w, h);
        maskCtx.fillStyle = '#fff';
        maskCtx.beginPath();
        maskCtx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2);
        maskCtx.fill();

        // 2. Feather the mask via Gaussian blur
        const maskData = maskCtx.getImageData(0, 0, w, h);
        const featheredData = separableGaussianBlur(maskData.data, w, h, FEATHER_RADIUS);
        const featherCanvas = createCanvas(w, h);
        const featherCtx = featherCanvas.getContext('2d');
        featherCtx.putImageData(new ImageData(featheredData, w, h), 0, 0);

        // 3. Blurred image clipped by feathered mask
        const patchCanvas = createCanvas(w, h);
        const patchCtx = patchCanvas.getContext('2d');
        patchCtx.drawImage(blurredCanvas, 0, 0);
        patchCtx.globalCompositeOperation = 'destination-in';
        patchCtx.drawImage(featherCanvas, 0, 0);
        patchCtx.globalCompositeOperation = 'source-over';

        // 4. Composite onto the main canvas
        ctx.drawImage(patchCanvas, 0, 0);

        console.log(`   Face ${i + 1}: score=${det.detection.score.toFixed(2)}  box=[${Math.round(box.x)},${Math.round(box.y)},${Math.round(box.width)},${Math.round(box.height)}]`);
    });

    // ── 6. Save result ────────────────────────────────────────────────────────
    const buf = canvas.toBuffer('image/png');
    fs.writeFileSync(outputPath, buf);
    console.log(`\n✅ All faces blurred`);
    console.log(`💾 Saved: ${outputPath}\n`);
}

// ─── CLI ──────────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
if (args.length < 1) {
    console.error('Usage: node blur-faces.js <input-image> [output-image]');
    process.exit(1);
}

const inputFile = args[0];
const outputFile = args[1] || `faceshield-${path.basename(inputFile, path.extname(inputFile))}.png`;

if (!fs.existsSync(inputFile)) {
    console.error(`❌ File not found: ${inputFile}`);
    process.exit(1);
}

blurFaces(inputFile, outputFile).catch(err => {
    console.error('❌ Error:', err.message || err);
    process.exit(1);
});
