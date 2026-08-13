/*
 * pipeline.js — the qi26_25 inference pipeline, in the browser.
 *
 * This is a direct port of integration/pipeline.py and integration/features.py.
 * It loads model.json (exported by web/export_model.py) so the learned weights
 * and the filter unitary are the exact ones the report measured — nothing is
 * re-fitted or approximated here.
 *
 * Verified against the Python implementation by web/verify_parity.py, which
 * runs both over the same images and compares stage by stage.
 *
 * WHY NO QUANTUM SIMULATOR IS NEEDED
 * ----------------------------------
 * Two quantum stages, both reducible to small dense linear algebra:
 *
 *  1. Quanvolutional layer. The filter is a FIXED circuit, so it is one 16x16
 *     unitary U. The RY angle encoding produces a product state, so the input
 *     statevector is a Kronecker product of four single-qubit states. Applying
 *     the filter is U @ psi. This is precisely what Aer computes.
 *
 *  2. Grover search. The comparator oracle was verified in Python to have ZERO
 *     uncomputation leakage — the window register returns exactly to |0>, so
 *     the position register is unentangled from it after each oracle call.
 *     That measurement is what licenses simulating the position register alone
 *     (2^n_pos amplitudes, n_pos <= 4) instead of the full 14-qubit space. The
 *     reduced dynamics are exact, not an approximation, and the leakage check
 *     is the evidence.
 */

export const Pipeline = (() => {
  let M = null; // model.json

  async function load(url = "./model.json") {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`could not load ${url} (${r.status})`);
    M = await r.json();
    return M.meta;
  }

  // ---------------------------------------------------------------- utils

  function toGray(imgData) {
    const { width: w, height: h, data } = imgData;
    const g = new Float64Array(w * h);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      // Pillow's exact integer luma (ImagingConvertL):
      //   L = (R*19595 + G*38470 + B*7471 + 0x8000) >> 16
      // The float form 0.299R + 0.587G + 0.114B is NOT equivalent: for a grey
      // pixel it returns v * 1.0000000000000002, a 3e-14 error. That sounds
      // harmless, but the resampler multiplies by 2^22 before flooring, so it
      // can tip a value across an integer boundary and change the crop.
      g[p] = (data[i] * 19595 + data[i + 1] * 38470 + data[i + 2] * 7471 + 0x8000) >> 16;
    }
    return { w, h, g };
  }

  function medianFilter3(img) {
    const { w, h, g } = img;
    const out = new Float64Array(w * h);
    const win = new Float64Array(9);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const yy = Math.min(h - 1, Math.max(0, y + dy));
            const xx = Math.min(w - 1, Math.max(0, x + dx));
            win[n++] = g[yy * w + xx];
          }
        }
        const s = Array.prototype.slice.call(win, 0, n).sort((a, b) => a - b);
        out[y * w + x] = s[4];
      }
    }
    return { w, h, g: out };
  }

  /** Otsu threshold, same algorithm as pipeline.binarize. */
  function otsu(img) {
    const { g } = img;
    const hist = new Float64Array(256);
    for (let i = 0; i < g.length; i++) hist[Math.min(255, Math.max(0, Math.round(g[i])))]++;
    const total = g.length;
    let sumAll = 0;
    for (let t = 0; t < 256; t++) sumAll += t * hist[t];
    let sumB = 0, wB = 0, best = 128, bestVar = -1;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      const wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB, mF = (sumAll - sumB) / wF;
      const v = wB * wF * (mB - mF) * (mB - mF);
      if (v > bestVar) { bestVar = v; best = t; }
    }
    return best;
  }

  /** Connected-component speckle removal (scipy.ndimage.label equivalent). */
  function cleanInk(ink, w, h, minComponent = 6) {
    const lab = new Int32Array(w * h).fill(0);
    const sizes = [0];
    let cur = 0;
    const stack = [];
    for (let i = 0; i < w * h; i++) {
      if (!ink[i] || lab[i]) continue;
      cur++; let n = 0;
      stack.push(i);
      lab[i] = cur;
      while (stack.length) {
        const p = stack.pop(); n++;
        const px = p % w, py = (p / w) | 0;
        // 4-connectivity, NOT 8. scipy.ndimage.label defaults to a cross-shaped
        // structuring element, and the pipeline relies on that default. Using
        // 8-connectivity here merges diagonally-touching speckle into glyph
        // components, which changes which blobs survive the size filter and
        // shifted one character's bounding box by a pixel.
        const nb = [[0, -1], [0, 1], [-1, 0], [1, 0]];
        for (const [dx, dy] of nb) {
          const nx = px + dx, ny = py + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q = ny * w + nx;
          if (ink[q] && !lab[q]) { lab[q] = cur; stack.push(q); }
        }
      }
      sizes.push(n);
    }
    const out = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) {
      if (lab[i] && sizes[lab[i]] >= minComponent) out[i] = 1;
    }
    return out;
  }

  function preprocess(img) {
    const sm = medianFilter3(img);
    const t = otsu(sm);
    const { w, h, g } = sm;
    const ink = new Uint8Array(w * h);
    for (let i = 0; i < g.length; i++) ink[i] = g[i] <= t ? 1 : 0;
    return { ink: cleanInk(ink, w, h), w, h, threshold: t };
  }

  // ---------------------------------------------------------- segmentation

  function segment(img) {
    const { ink, w, h } = preprocess(img);

    const rowThresh = Math.max(2, Math.floor(0.008 * w));
    const rowInk = new Int32Array(h);
    for (let y = 0; y < h; y++) {
      let s = 0;
      for (let x = 0; x < w; x++) s += ink[y * w + x];
      rowInk[y] = s;
    }

    const lines = [];
    let start = null;
    for (let y = 0; y < h; y++) {
      const a = rowInk[y] > rowThresh;
      if (a && start === null) start = y;
      else if (!a && start !== null) {
        if (y - start >= 4) lines.push([start, y]);
        start = null;
      }
    }
    if (start !== null && h - start >= 4) lines.push([start, h]);

    let boxes = [];
    lines.forEach(([y0, y1], li) => {
      const colInk = new Int32Array(w);
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let y = y0; y < y1; y++) s += ink[y * w + x];
        colInk[x] = s;
      }
      let cs = null;
      for (let x = 0; x <= w; x++) {
        const a = x < w && colInk[x] > 0;
        if (a && cs === null) cs = x;
        else if (!a && cs !== null) {
          let tot = 0;
          for (let xx = cs; xx < x; xx++) tot += colInk[xx];
          if (tot >= 4 && x - cs >= 2) {
            let ty = y1, by = y0;
            for (let y = y0; y < y1; y++) {
              for (let xx = cs; xx < x; xx++) {
                if (ink[y * w + xx]) { if (y < ty) ty = y; if (y + 1 > by) by = y + 1; }
              }
            }
            boxes.push({ line: li, x0: cs, y0: ty, x1: x, y1: by });
          }
          cs = null;
        }
      }
    });

    return { boxes: splitWide(boxes, ink, w), ink, w, h };
  }

  /** Valley-based splitting of merged glyphs — mirrors split_wide_boxes. */
  function splitWide(boxes, ink, w, ratio = 1.8, valleyFrac = 0.45) {
    if (boxes.length < 3) return boxes;
    const widths = boxes.map(b => b.x1 - b.x0).sort((a, b) => a - b);
    const ref = widths[Math.floor(0.6 * (widths.length - 1))];
    if (ref <= 0) return boxes;

    const out = [];
    for (const b of boxes) {
      const bw = b.x1 - b.x0;
      if (bw <= ratio * ref) { out.push(b); continue; }

      const col = new Float64Array(bw);
      for (let x = 0; x < bw; x++) {
        let s = 0;
        for (let y = b.y0; y < b.y1; y++) s += ink[y * w + (b.x0 + x)];
        col[x] = s;
      }
      const mean = col.reduce((a, c) => a + c, 0) / bw;
      if (bw < 6 || mean <= 0) { out.push(b); continue; }

      const margin = Math.max(2, Math.floor(0.25 * ref));
      const cuts = [];
      for (let c = margin; c < bw - margin; c++) {
        let localMin = Infinity;
        for (let k = Math.max(0, c - 2); k < Math.min(bw, c + 3); k++) {
          localMin = Math.min(localMin, col[k]);
        }
        if (col[c] < valleyFrac * mean && col[c] <= localMin) {
          if (!cuts.length || c - cuts[cuts.length - 1] >= margin) cuts.push(c);
        }
      }
      if (!cuts.length) { out.push(b); continue; }

      let prev = 0;
      for (const c of cuts.concat([bw])) {
        if (c - prev >= 2) {
          out.push({ line: b.line, x0: b.x0 + prev, y0: b.y0, x1: b.x0 + c, y1: b.y1 });
        }
        prev = c;
      }
    }
    return out;
  }

  // ------------------------------------------------------------- cropping

  /** Pad to square then resize — mirrors features.square_pad_resize. */
  function squarePadResize(img, box, size, pad = 1) {
    const { w, h, g } = img;
    const x0 = Math.max(0, box.x0 - pad), x1 = Math.min(w, box.x1 + pad);
    const y0 = Math.max(0, box.y0 - pad), y1 = Math.min(h, box.y1 + pad);
    const bw = x1 - x0, bh = y1 - y0;
    const side = Math.max(bw, bh);

    const canvas = new Float64Array(side * side).fill(255);
    const oy = ((side - bh) / 2) | 0, ox = ((side - bw) / 2) | 0;
    for (let y = 0; y < bh; y++) {
      for (let x = 0; x < bw; x++) {
        canvas[(oy + y) * side + (ox + x)] = g[(y0 + y) * w + (x0 + x)];
      }
    }

    return resampleBilinear(canvas, side, side, size, size);
  }

  /**
   * PIL-compatible BILINEAR resampling.
   *
   * Naive 2x2 bilinear sampling is NOT what PIL does and produces visibly
   * different crops when downscaling: Pillow applies a triangle filter whose
   * support is scaled by the reduction factor, i.e. antialiasing. A 19x19 glyph
   * reduced to 8x8 therefore averages over a ~2.4px window per output pixel,
   * while naive sampling reads only the nearest 2x2 and aliases away most of
   * the stroke.
   *
   * This cost real accuracy before it was fixed: with naive sampling the
   * browser decoded 'QWORLO RESEARCT' where Python decoded 'QWORLD RESEARCH',
   * because the crops fed to the classifier were not the crops it was trained
   * on. Ported from Pillow's resample.c (ImagingResampleHorizontal + triangle
   * filter) so the two agree exactly.
   */
  /**
   * Pillow-exact BILINEAR resampling for 8-bit images.
   *
   * Three details matter and each cost a debugging round to find:
   *
   *  1. Pillow scales the triangle filter's support by the reduction factor
   *     (antialiasing). Naive 2x2 sampling aliases away most of a glyph stroke
   *     when reducing 19px to 8px.
   *  2. Window bounds use C integer truncation, not Math.ceil.
   *  3. Pillow accumulates in FIXED-POINT INTEGER arithmetic, quantising the
   *     normalised coefficients to 22-bit integers, and rounds to uint8 after
   *     each 1D pass. Float accumulation left 14 of 46 crops off by exactly 1
   *     grey level, which was enough to flip one borderline glyph.
   *
   * Ported from Pillow's resample.c (precompute_coeffs, normalize_coeffs_8bpc,
   * ImagingResampleHorizontal_8bpc). Verified bit-exact by web/verify_parity.py.
   */
  const PRECISION_BITS = 32 - 8 - 2;   // 22, as in Pillow
  const FIXED_ONE = 1 << PRECISION_BITS;
  const ROUND_OFFSET = 1 << (PRECISION_BITS - 1);

  function clip8fixed(acc) {
    const v = Math.floor(acc / FIXED_ONE);   // arithmetic shift right
    return v < 0 ? 0 : v > 255 ? 255 : v;
  }

  function resample1D(src, srcW, srcH, dstSize, horizontal) {
    const inSize = horizontal ? srcW : srcH;
    const scale = inSize / dstSize;
    const filterScale = Math.max(scale, 1.0);
    const support = 1.0 * filterScale;
    const invFs = 1.0 / filterScale;

    const outW = horizontal ? dstSize : srcW;
    const outH = horizontal ? srcH : dstSize;
    const out = new Float64Array(outW * outH);

    for (let i = 0; i < dstSize; i++) {
      const center = (i + 0.5) * scale;
      let xmin = Math.trunc(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5);
      if (xmax > inSize) xmax = inSize;
      const n = Math.max(xmax - xmin, 1);

      // 1. float coefficients, normalised
      const kf = new Float64Array(n);
      let ww = 0;
      for (let x = 0; x < n; x++) {
        const t = Math.abs((x + xmin - center + 0.5) * invFs);
        const w = t < 1.0 ? 1.0 - t : 0.0;
        kf[x] = w; ww += w;
      }
      if (ww !== 0) for (let x = 0; x < n; x++) kf[x] /= ww;

      // 2. quantise to fixed point, rounding away from zero (Pillow)
      const k = new Int32Array(n);
      for (let x = 0; x < n; x++) {
        k[x] = kf[x] < 0
          ? Math.trunc(-0.5 + kf[x] * FIXED_ONE)
          : Math.trunc(0.5 + kf[x] * FIXED_ONE);
      }

      // 3. integer accumulate with rounding offset, then clip to uint8
      if (horizontal) {
        for (let y = 0; y < srcH; y++) {
          let acc = ROUND_OFFSET;
          for (let x = 0; x < n; x++) acc += src[y * srcW + (xmin + x)] * k[x];
          out[y * dstSize + i] = clip8fixed(acc);
        }
      } else {
        for (let x = 0; x < srcW; x++) {
          let acc = ROUND_OFFSET;
          for (let y = 0; y < n; y++) acc += src[(xmin + y) * srcW + x] * k[y];
          out[i * srcW + x] = clip8fixed(acc);
        }
      }
    }
    return out;
  }

  function resampleBilinear(src, srcW, srcH, dstW, dstH) {
    const h = resample1D(src, srcW, srcH, dstW, true);
    return resample1D(h, dstW, srcH, dstH, false);
  }

  /** Per-crop min-max stretch — mirrors features.normalize_crops. */
  function normalizeCrop(c) {
    let lo = Infinity, hi = -Infinity;
    for (const v of c) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = Math.max(hi - lo, 1e-6);
    const out = new Float64Array(c.length);
    for (let i = 0; i < c.length; i++) out[i] = ((c[i] - lo) / span) * 255;
    return out;
  }

  // ------------------------------------------- quanvolutional feature stage

  /**
   * One 2x2 patch -> 10 features (4 marginals + 6 <Z_i Z_j> correlations).
   * Mirrors features.quanv_patch_probs with correlations=True.
   */
  function quanvPatch(patch) {
    const nq = M.constants.n_qubits;          // 4
    const dim = 1 << nq;                       // 16

    // RY(pi * v/255) on |0> gives [cos(t/2), sin(t/2)]; joint state is the
    // Kronecker product. Qiskit is little-endian, so qubit 0 is the LEAST
    // significant index — the product is built from the highest qubit down.
    let state = new Float64Array([1]);
    for (let q = nq - 1; q >= 0; q--) {
      const t = Math.PI * (patch[q] / 255);
      const c = Math.cos(t / 2), s = Math.sin(t / 2);
      const next = new Float64Array(state.length * 2);
      for (let i = 0; i < state.length; i++) {
        next[i * 2] = state[i] * c;
        next[i * 2 + 1] = state[i] * s;
      }
      state = next;
    }

    // Apply U (complex). Input amplitudes are real, so only U's columns mix.
    const Ur = M.filter_unitary_real, Ui = M.filter_unitary_imag;
    const probs = new Float64Array(dim);
    for (let r = 0; r < dim; r++) {
      let re = 0, im = 0;
      for (let c = 0; c < dim; c++) {
        re += Ur[r][c] * state[c];
        im += Ui[r][c] * state[c];
      }
      probs[r] = re * re + im * im;
    }

    const out = new Float64Array(nq + (nq * (nq - 1)) / 2);
    for (let q = 0; q < nq; q++) {
      let p = 0;
      for (let i = 0; i < dim; i++) if ((i >> q) & 1) p += probs[i];
      out[q] = p;
    }
    let k = nq;
    for (let i = 0; i < nq; i++) {
      for (let j = i + 1; j < nq; j++) {
        let s = 0;
        for (let b = 0; b < dim; b++) {
          const zi = ((b >> i) & 1) ? -1 : 1;
          const zj = ((b >> j) & 1) ? -1 : 1;
          s += probs[b] * zi * zj;
        }
        out[k++] = s;
      }
    }
    return out;
  }

  /** Full 8x8 crop -> 160-dim feature vector (16 patches x 10). */
  function quanvFeatures(crop) {
    const cs = M.constants.crop_size, ps = M.constants.patch_size, st = M.constants.stride;
    const outH = ((cs - ps) / st | 0) + 1, outW = outH;
    const per = M.constants.n_qubits + (M.constants.n_qubits * (M.constants.n_qubits - 1)) / 2;
    const feat = new Float64Array(outH * outW * per);
    const patch = new Float64Array(ps * ps);
    let k = 0;
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let n = 0;
        for (let y = 0; y < ps; y++) {
          for (let x = 0; x < ps; x++) {
            patch[n++] = crop[(oy * st + y) * cs + (ox * st + x)];
          }
        }
        feat.set(quanvPatch(patch), k * per);
        k++;
      }
    }
    return feat;
  }

  function classify(feat) {
    const coef = M.ocr_coef, b = M.ocr_intercept;
    let best = 0, bestV = -Infinity;
    const logits = new Float64Array(coef.length);
    for (let c = 0; c < coef.length; c++) {
      let s = b[c];
      const row = coef[c];
      for (let i = 0; i < feat.length; i++) s += row[i] * feat[i];
      logits[c] = s;
      if (s > bestV) { bestV = s; best = c; }
    }
    // softmax confidence for the display
    let mx = bestV, sum = 0;
    for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - mx);
    return { char: M.classes[best], confidence: 1 / sum, index: best };
  }

  // ------------------------------------------------------- text assembly

  function assembleText(boxes, chars, spaceGap = 1.6) {
    const lines = new Map();
    boxes.forEach((b, i) => {
      if (!lines.has(b.line)) lines.set(b.line, []);
      lines.get(b.line).push({ x0: b.x0, x1: b.x1, ch: chars[i] });
    });

    const out = [];
    for (const li of [...lines.keys()].sort((a, b) => a - b)) {
      const items = lines.get(li).sort((a, b) => a.x0 - b.x0);
      if (!items.length) continue;
      const gaps = [];
      for (let k = 1; k < items.length; k++) gaps.push(items[k].x0 - items[k - 1].x1);
      let thresh = Infinity;
      if (gaps.length) {
        const sorted = [...gaps].sort((a, b) => a - b);
        const med = sorted[Math.floor(sorted.length / 2)];
        const mean = gaps.reduce((a, c) => a + c, 0) / gaps.length;
        const std = Math.sqrt(gaps.reduce((a, c) => a + (c - mean) ** 2, 0) / gaps.length);
        thresh = Math.max(med + spaceGap * (std + 1e-6), med * 2, 3);
      }
      let s = items[0].ch;
      for (let k = 1; k < items.length; k++) {
        if (items[k].x0 - items[k - 1].x1 > thresh) s += " ";
        s += items[k].ch;
      }
      out.push(s);
    }
    return out.join("\n");
  }

  /**
   * Layout-driven ID extraction — mirrors pipeline._locate_id_field.
   * Scores per TOKEN, not per line: when OCR drops the colon in
   * "DOCUMENT ID: QI96-3898", line-level scoring harvests the D/C/E of
   * "DOCUMENT" as hex digits and returns "DCED963898" instead of "963898",
   * which shifts every position the Grover stage reports.
   */
  function locateIdField(text) {
    let best = "", bestScore = -1;
    for (const line of text.toUpperCase().split("\n")) {
      for (const token of line.replace(/:/g, " ").split(/\s+/)) {
        const hex = [...token].filter(c => "0123456789ABCDEF".includes(c)).join("");
        if (hex.length < 3) continue;
        const digits = [...hex].filter(c => c >= "0" && c <= "9").length;
        if (digits === 0) continue;   // pure-letter runs are words, not IDs
        // Length dominates, digit density modulates — see pipeline.py.
        const score = hex.length * (0.5 + 0.5 * digits / hex.length);
        if (score > bestScore) { best = hex; bestScore = score; }
      }
    }
    return best.slice(0, 16);
  }

  // -------------------------------------------------------- Grover search

  /**
   * Grover over the position register.
   *
   * Exact because the Python comparator oracle was measured to have zero
   * uncomputation leakage: the window register returns to |0>, leaving the
   * position register in a pure state of its own. The oracle's action on that
   * subspace is exactly a sign flip on matching positions, and the diffusion
   * operator is the standard reflection about the mean.
   */
  function groverSearch(text, pattern) {
    const M_ = pattern.length;
    const nCand = text.length - M_ + 1;
    if (nCand < 1) return { status: "pattern longer than text" };

    const nPos = Math.max(1, Math.ceil(Math.log2(nCand)));
    const dim = 1 << nPos;
    const truth = [];
    for (let i = 0; i < nCand; i++) if (text.slice(i, i + M_) === pattern) truth.push(i);
    if (!truth.length) return { status: "pattern absent", searchable: text, nPos };

    const iterations = Math.max(1, Math.floor((Math.PI / 4) * Math.sqrt(dim / truth.length)));
    let amp = new Float64Array(dim).fill(1 / Math.sqrt(dim));
    const history = [Array.from(amp)];

    for (let it = 0; it < iterations; it++) {
      for (const t of truth) amp[t] = -amp[t];          // oracle
      const mean = amp.reduce((a, c) => a + c, 0) / dim; // diffusion
      for (let i = 0; i < dim; i++) amp[i] = 2 * mean - amp[i];
      history.push(Array.from(amp));
    }

    const probs = Array.from(amp, a => a * a);
    let best = 0;
    probs.forEach((p, i) => { if (p > probs[best]) best = i; });

    return {
      status: "ok", searchable: text, pattern, truth, best,
      correct: truth.includes(best), confidence: probs[best],
      iterations, nPos, probs, history, nCand,
      qubitsFullCircuit: nPos + M_ * 5,
    };
  }

  // ------------------------------------------------------------ full run

  /** Run everything, reporting progress so the UI can animate stage by stage. */
  async function run(imgData, pattern, onStage = () => {}) {
    if (!M) throw new Error("call Pipeline.load() first");
    const t0 = performance.now();

    const img = toGray(imgData);
    onStage("input", { w: img.w, h: img.h });

    const seg = segment(img);
    onStage("segment", { boxes: seg.boxes, ink: seg.ink, w: seg.w, h: seg.h });
    await tick();

    const crops = seg.boxes.map(b => squarePadResize(img, b, M.constants.crop_size));
    const norm = crops.map(normalizeCrop);
    onStage("crops", { crops: norm });
    await tick();

    const feats = norm.map(quanvFeatures);
    onStage("features", { features: feats });
    await tick();

    const preds = feats.map(classify);
    const text = assembleText(seg.boxes, preds.map(p => p.char));
    onStage("classify", { predictions: preds, text });
    await tick();

    const field = locateIdField(text);
    const grover = pattern ? groverSearch(field, pattern.toUpperCase()) : null;
    onStage("grover", { field, grover });

    return {
      boxes: seg.boxes, crops: norm, features: feats, predictions: preds,
      text, field, grover, ms: performance.now() - t0,
    };
  }

  const tick = () => new Promise(r => requestAnimationFrame(() => r()));

  return { load, run, segment, quanvFeatures, quanvPatch, classify,
           groverSearch, locateIdField, assembleText, squarePadResize,
           normalizeCrop, toGray, otsu, get model() { return M; } };
})();
