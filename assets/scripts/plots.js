// ============================================================
// DATA
// ============================================================

// Seeded PRNG for reproducible scatter data
function mulberry32(a) {
    return function() {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        var t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// Generate prevalence data with scatter noise (like the paper)
function generatePrevalenceData() {
    const months = [];
    const labels = [];
    const aiGenerated = [];
    const aiGeneratedOrAssisted = [];
    const xIndices = [];

    const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const rand = mulberry32(314);

    // Aug 2022 (month 0) to May 2025 (month 33)
    for (let i = 0; i <= 33; i++) {
        const monthIdx = (8 + i - 1) % 12;
        const year = 2022 + Math.floor((7 + i) / 12);
        const m = monthIdx + 1;
        months.push(`${year}-${String(m).padStart(2, '0')}-15`);
        labels.push(`${monthNames[monthIdx]} ${year}`);
        xIndices.push(i);

        // Logistic growth for AI-gen+assisted with scatter noise
        const assisted = 0.38 / (1 + Math.exp(-0.20 * (i - 21)));
        const generated = 0.155 / (1 + Math.exp(-0.20 * (i - 22)));

        // Add noise to scatter points (proportional to value, more realistic)
        const noiseAssisted = (rand() - 0.5) * 0.04 * Math.max(0.3, assisted / 0.38);
        const noiseGenerated = (rand() - 0.5) * 0.025 * Math.max(0.3, generated / 0.155);

        aiGeneratedOrAssisted.push(Math.round(Math.max(0, assisted + noiseAssisted) * 1000) / 10);
        aiGenerated.push(Math.round(Math.max(0, generated + noiseGenerated) * 1000) / 10);
    }

    return { months, labels, aiGenerated, aiGeneratedOrAssisted, xIndices };
}

// Fit a polynomial of given degree and return smooth curve + confidence band
function polyFit(xArr, yArr, degree, nSmooth) {
    const n = xArr.length;
    degree = Math.min(degree, n - 1);

    // Normalize x to [-1, 1] for numerical stability
    const xMin = Math.min(...xArr);
    const xMax = Math.max(...xArr);
    const xRange = xMax - xMin || 1;
    const xNorm = xArr.map(x => (x - xMin) / xRange * 2 - 1);

    // Build Vandermonde matrix and solve via normal equations
    // coeffs[0] = highest degree
    const coeffs = polyFitCoeffs(xNorm, yArr, degree);

    // Evaluate smooth curve
    const xSmooth = [];
    for (let i = 0; i < nSmooth; i++) {
        xSmooth.push(xMin + (xMax - xMin) * i / (nSmooth - 1));
    }
    const xSmoothNorm = xSmooth.map(x => (x - xMin) / xRange * 2 - 1);

    const ySmooth = xSmoothNorm.map(x => polyEval(coeffs, x));

    // Residual std error for confidence band
    const yPred = xNorm.map(x => polyEval(coeffs, x));
    const residuals = yArr.map((y, i) => y - yPred[i]);
    const stdErr = Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / n);

    const yUpper = ySmooth.map(y => Math.max(0, Math.min(y + stdErr, 100)));
    const yLower = ySmooth.map(y => Math.max(0, Math.min(y - stdErr, 100)));
    const yClamped = ySmooth.map(y => Math.max(0, Math.min(y, 100)));

    return { xSmooth, ySmooth: yClamped, yUpper, yLower };
}

function polyEval(coeffs, x) {
    let result = 0;
    for (let i = 0; i < coeffs.length; i++) {
        result = result * x + coeffs[i];
    }
    return result;
}

// Simple polynomial fitting using least squares (normal equations)
function polyFitCoeffs(x, y, degree) {
    const n = x.length;
    const m = degree + 1;

    // Build Vandermonde matrix V[i][j] = x[i]^(degree-j)
    const V = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < m; j++) {
            row.push(Math.pow(x[i], degree - j));
        }
        V.push(row);
    }

    // Compute V^T * V
    const VtV = [];
    for (let i = 0; i < m; i++) {
        VtV.push([]);
        for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) sum += V[k][i] * V[k][j];
            VtV[i].push(sum);
        }
    }

    // Compute V^T * y
    const Vty = [];
    for (let i = 0; i < m; i++) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += V[k][i] * y[k];
        Vty.push(sum);
    }

    // Solve via Gaussian elimination
    return gaussianSolve(VtV, Vty);
}

function gaussianSolve(A, b) {
    const n = A.length;
    // Augment
    const M = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < n; col++) {
        // Partial pivoting
        let maxRow = col;
        for (let row = col + 1; row < n; row++) {
            if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
        }
        [M[col], M[maxRow]] = [M[maxRow], M[col]];

        const pivot = M[col][col];
        if (Math.abs(pivot) < 1e-12) continue;

        for (let j = col; j <= n; j++) M[col][j] /= pivot;
        for (let row = 0; row < n; row++) {
            if (row === col) continue;
            const factor = M[row][col];
            for (let j = col; j <= n; j++) M[row][j] -= factor * M[col][j];
        }
    }

    return M.map(row => row[n]);
}

// Generate correlated scatter data for a hypothesis
function generateScatterData(rho, seed, yMean, yStd, yLabel) {
    const rand = mulberry32(seed);
    const n = 36;
    const points = [];

    // Generate x values (AI likelihood scores, increasing over time)
    const xVals = [];
    const noise1 = [];
    const noise2 = [];
    for (let i = 0; i < n; i++) {
        // Base x follows logistic growth
        const base = 0.35 / (1 + Math.exp(-0.22 * (i - 18)));
        xVals.push(base + (rand() - 0.5) * 0.03);
        noise1.push(gaussianRand(rand));
        noise2.push(gaussianRand(rand));
    }

    // Create correlated y using Cholesky-like approach
    for (let i = 0; i < n; i++) {
        const x = xVals[i];
        // y = rho * (standardized x) + sqrt(1-rho^2) * independent noise
        const xNorm = (x - 0.15) / 0.10;
        const yNorm = rho * xNorm + Math.sqrt(1 - rho * rho) * noise1[i];
        const y = yMean + yStd * yNorm;
        points.push({ x: Math.round(x * 1000) / 1000, y: Math.round(y * 10000) / 10000 });
    }

    return points;
}

// Generate time-series data for dual-axis plot
function generateTimeSeriesData(rho, seed, yMean, yStd) {
    const rand = mulberry32(seed + 1000);
    const n = 36;
    const months = [];
    const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

    const aiLikelihood = [];
    const signal = [];

    for (let i = 0; i < n; i++) {
        const monthIdx = (8 + i - 1) % 12;
        const year = 2022 + Math.floor((7 + i) / 12);
        const m = monthIdx + 1;
        months.push(`${year}-${String(m).padStart(2, '0')}-15`);

        // AI likelihood follows logistic growth
        const aiBase = 0.35 / (1 + Math.exp(-0.22 * (i - 18)));
        const ai = aiBase + (rand() - 0.5) * 0.02;
        aiLikelihood.push(Math.round(ai * 1000) / 1000);

        // Signal is correlated with AI likelihood via rho
        const aiNorm = (ai - 0.15) / 0.10;
        const yNorm = rho * aiNorm + Math.sqrt(1 - rho * rho) * gaussianRand(rand);
        const y = yMean + yStd * yNorm;
        signal.push(Math.round(y * 10000) / 10000);
    }

    return { months, aiLikelihood, signal };
}

function gaussianRand(rand) {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Survey data from the paper (approximate counts matching reported percentages)
const SURVEY_DATA = {
    h1: {
        name: "Semantic Contraction",
        statement: "As AI text becomes more common on the internet, the range of unique ideas and diverse viewpoints shrinks.",
        n: 303,
        overall: { SD: 8, D: 24, SoD: 51, SoA: 82, A: 75, SA: 28, N: 35 },
        byUsage: {
            "Never":  { SD: 1, D: 3, SoD: 5, N: 3, SoA: 10, A: 9, SA: 3 },
            "Monthly": { SD: 1, D: 5, SoD: 9, N: 6, SoA: 16, A: 12, SA: 3 },
            "Weekly":  { SD: 3, D: 8, SoD: 18, N: 12, SoA: 26, A: 24, SA: 10 },
            "Daily":   { SD: 3, D: 8, SoD: 19, N: 14, SoA: 30, A: 30, SA: 12 }
        },
        byView: {
            "Negative": { SD: 1, D: 4, SoD: 11, N: 8, SoA: 30, A: 38, SA: 20 },
            "Neutral":  { SD: 1, D: 4, SoD: 8, N: 10, SoA: 10, A: 4, SA: 1 },
            "Positive": { SD: 6, D: 16, SoD: 32, N: 17, SoA: 42, A: 33, SA: 7 }
        },
        rho: 0.47, p: 0.004, confirmed: true,
        xLabel: "Aggregate AI Likelihood Score",
        yLabel: "Avg. Pairwise Cosine Similarity",
        scatterSeed: 42, yMean: 0.058, yStd: 0.008
    },
    h2: {
        name: "Truth Decay",
        statement: "As AI content becomes more common on the internet, I am encountering factually incorrect information and hallucinations more frequently.",
        n: 303,
        overall: { SD: 6, D: 17, SoD: 20, SoA: 74, A: 80, SA: 48, N: 43 },
        byUsage: {
            "Never":  { SD: 0, D: 2, SoD: 2, N: 4, SoA: 8, A: 10, SA: 8 },
            "Monthly": { SD: 1, D: 3, SoD: 4, N: 7, SoA: 15, A: 15, SA: 7 },
            "Weekly":  { SD: 2, D: 5, SoD: 7, N: 14, SoA: 25, A: 26, SA: 16 },
            "Daily":   { SD: 3, D: 7, SoD: 7, N: 18, SoA: 26, A: 29, SA: 17 }
        },
        byView: {
            "Negative": { SD: 0, D: 3, SoD: 5, N: 10, SoA: 28, A: 38, SA: 28 },
            "Neutral":  { SD: 1, D: 2, SoD: 4, N: 12, SoA: 10, A: 6, SA: 3 },
            "Positive": { SD: 5, D: 12, SoD: 11, N: 21, SoA: 36, A: 36, SA: 17 }
        },
        rho: -0.19, p: 0.27, confirmed: false,
        xLabel: "Aggregate AI Likelihood Score",
        yLabel: "Factual Error Rate",
        scatterSeed: 99, yMean: 0.12, yStd: 0.03
    },
    h3: {
        name: "Positivity Shift",
        statement: "As AI content becomes more common on the internet, online writing feels increasingly sanitized and artificially cheerful.",
        n: 303,
        overall: { SD: 5, D: 14, SoD: 20, SoA: 80, A: 78, SA: 60, N: 46 },
        byUsage: {
            "Never":  { SD: 0, D: 1, SoD: 2, N: 4, SoA: 9, A: 10, SA: 8 },
            "Monthly": { SD: 1, D: 2, SoD: 4, N: 8, SoA: 16, A: 14, SA: 7 },
            "Weekly":  { SD: 2, D: 5, SoD: 7, N: 15, SoA: 27, A: 25, SA: 20 },
            "Daily":   { SD: 2, D: 6, SoD: 7, N: 19, SoA: 28, A: 29, SA: 25 }
        },
        byView: {
            "Negative": { SD: 0, D: 2, SoD: 4, N: 10, SoA: 30, A: 35, SA: 31 },
            "Neutral":  { SD: 1, D: 2, SoD: 3, N: 14, SoA: 10, A: 5, SA: 3 },
            "Positive": { SD: 4, D: 10, SoD: 13, N: 22, SoA: 40, A: 38, SA: 26 }
        },
        rho: 0.56, p: 0.0003, confirmed: true,
        xLabel: "Aggregate AI Likelihood Score",
        yLabel: "Rate of Positive Documents",
        scatterSeed: 77, yMean: 0.42, yStd: 0.06
    },
    h4: {
        name: "Epistemic Islands",
        statement: "As AI content becomes more common on the internet, articles are increasingly providing answers without including links to external sources.",
        n: 301,
        overall: { SD: 8, D: 15, SoD: 12, SoA: 84, A: 77, SA: 40, N: 56 },
        byUsage: {
            "Never":  { SD: 1, D: 1, SoD: 1, N: 6, SoA: 10, A: 10, SA: 5 },
            "Monthly": { SD: 1, D: 3, SoD: 2, N: 10, SoA: 17, A: 14, SA: 5 },
            "Weekly":  { SD: 3, D: 5, SoD: 4, N: 18, SoA: 28, A: 25, SA: 15 },
            "Daily":   { SD: 3, D: 6, SoD: 5, N: 22, SoA: 29, A: 28, SA: 15 }
        },
        byView: {
            "Negative": { SD: 1, D: 2, SoD: 2, N: 12, SoA: 30, A: 35, SA: 22 },
            "Neutral":  { SD: 1, D: 2, SoD: 2, N: 18, SoA: 10, A: 6, SA: 3 },
            "Positive": { SD: 6, D: 11, SoD: 8, N: 26, SoA: 44, A: 36, SA: 15 }
        },
        rho: -0.12, p: 0.48, confirmed: false,
        xLabel: "Aggregate AI Likelihood Score",
        yLabel: "Outbound Link Density (per 1k words)",
        scatterSeed: 55, yMean: 8.5, yStd: 2.5
    },
    h5: {
        name: "Entropy Dilution",
        statement: "As AI content becomes more common on the internet, content is becoming significantly longer in word count while having lower semantic density.",
        n: 299,
        overall: { SD: 5, D: 19, SoD: 52, SoA: 82, A: 64, SA: 36, N: 41 },
        byUsage: {
            "Never":  { SD: 1, D: 2, SoD: 5, N: 4, SoA: 10, A: 8, SA: 4 },
            "Monthly": { SD: 1, D: 3, SoD: 10, N: 7, SoA: 15, A: 12, SA: 4 },
            "Weekly":  { SD: 1, D: 7, SoD: 17, N: 14, SoA: 27, A: 20, SA: 13 },
            "Daily":   { SD: 2, D: 7, SoD: 20, N: 16, SoA: 30, A: 24, SA: 15 }
        },
        byView: {
            "Negative": { SD: 0, D: 3, SoD: 10, N: 9, SoA: 30, A: 30, SA: 20 },
            "Neutral":  { SD: 1, D: 3, SoD: 8, N: 12, SoA: 10, A: 4, SA: 2 },
            "Positive": { SD: 4, D: 13, SoD: 34, N: 20, SoA: 42, A: 30, SA: 14 }
        },
        rho: -0.02, p: 0.89, confirmed: false,
        xLabel: "Aggregate AI Likelihood Score",
        yLabel: "Gzip Compression Ratio",
        scatterSeed: 33, yMean: 0.31, yStd: 0.02
    },
    h6: {
        name: "Stylistic Monoculture",
        statement: "As AI content becomes more common on the internet, distinct individual writing styles are disappearing in favor of a generic, uniform voice.",
        n: 301,
        overall: { SD: 3, D: 6, SoD: 14, SoA: 78, A: 84, SA: 88, N: 28 },
        byUsage: {
            "Never":  { SD: 0, D: 0, SoD: 1, N: 2, SoA: 8, A: 10, SA: 13 },
            "Monthly": { SD: 0, D: 1, SoD: 2, N: 5, SoA: 15, A: 16, SA: 13 },
            "Weekly":  { SD: 1, D: 2, SoD: 5, N: 9, SoA: 26, A: 28, SA: 30 },
            "Daily":   { SD: 2, D: 3, SoD: 6, N: 12, SoA: 29, A: 30, SA: 32 }
        },
        byView: {
            "Negative": { SD: 0, D: 1, SoD: 2, N: 5, SoA: 28, A: 38, SA: 38 },
            "Neutral":  { SD: 0, D: 1, SoD: 2, N: 10, SoA: 10, A: 8, SA: 7 },
            "Positive": { SD: 3, D: 4, SoD: 10, N: 13, SoA: 40, A: 38, SA: 43 }
        },
        rho: 0.24, p: 0.17, confirmed: false,
        xLabel: "Aggregate AI Likelihood Score",
        yLabel: "Avg. Pairwise Jaccard Similarity (3-gram)",
        scatterSeed: 11, yMean: 0.145, yStd: 0.012
    }
};

// ============================================================
// PLOTLY CONFIG & LAYOUT DEFAULTS
// ============================================================

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false
};

const FONT_FAMILY = '"Source Serif 4", Charter, Georgia, serif';
const MONO_FONT = '"Fira Code", monospace';

const COLORS = {
    aiGenerated: '#e74c3c',
    aiAssisted: '#8e44ad',
    confirmed: '#27ae60',
    notConfirmed: '#95a5a6',
    regression: '#2c3e50',
    scatter: '#3498db',
    likert: {
        SD: '#c0392b',
        D: '#e74c3c',
        SoD: '#f39c9c',
        N: '#bdc3c7',
        SoA: '#85c1e9',
        A: '#3498db',
        SA: '#1a5276'
    }
};

// ============================================================
// PLOT FUNCTIONS
// ============================================================

function plotPrevalence() {
    const data = generatePrevalenceData();
    const degree = 6;
    const nSmooth = 300;

    // Fit degree-6 polynomial for both series (matching paper)
    const fitAssisted = polyFit(data.xIndices, data.aiGeneratedOrAssisted, degree, nSmooth);
    const fitGenerated = polyFit(data.xIndices, data.aiGenerated, degree, nSmooth);

    // Map smooth x indices back to dates
    const smoothMonths = fitAssisted.xSmooth.map(xi => {
        const monthIdx = (8 + Math.round(xi) - 1) % 12;
        const year = 2022 + Math.floor((7 + xi) / 12);
        const m = ((8 + xi - 1) % 12) + 1;
        const mInt = Math.floor(m) + 1;
        const day = Math.round((m - Math.floor(m)) * 28) + 1;
        // Interpolate date
        const frac = xi;
        const baseMonth = Math.floor(frac);
        const remainder = frac - baseMonth;
        if (baseMonth >= 0 && baseMonth < data.months.length - 1) {
            const d0 = new Date(data.months[baseMonth]);
            const d1 = new Date(data.months[Math.min(baseMonth + 1, data.months.length - 1)]);
            return new Date(d0.getTime() + remainder * (d1.getTime() - d0.getTime())).toISOString().slice(0, 10);
        } else if (baseMonth >= data.months.length - 1) {
            return data.months[data.months.length - 1];
        }
        return data.months[0];
    });

    // Scatter points for raw data
    const scatterAssisted = {
        x: data.months,
        y: data.aiGeneratedOrAssisted,
        name: '',
        type: 'scatter',
        mode: 'markers',
        marker: { color: COLORS.aiAssisted, size: 6, opacity: 0.4, symbol: 'square' },
        hovertemplate: '<b>%{text}</b><br>AI-Gen. or Assisted: %{y:.1f}%<extra></extra>',
        text: data.labels,
        showlegend: false
    };

    const scatterGenerated = {
        x: data.months,
        y: data.aiGenerated,
        name: '',
        type: 'scatter',
        mode: 'markers',
        marker: { color: COLORS.aiGenerated, size: 6, opacity: 0.4, symbol: 'circle' },
        hovertemplate: '<b>%{text}</b><br>Fully AI-Generated: %{y:.1f}%<extra></extra>',
        text: data.labels,
        showlegend: false
    };

    // Confidence band for assisted
    const bandAssisted = {
        x: [...smoothMonths, ...smoothMonths.slice().reverse()],
        y: [...fitAssisted.yUpper, ...fitAssisted.yLower.slice().reverse()],
        fill: 'toself',
        fillcolor: 'rgba(142, 68, 173, 0.15)',
        line: { color: 'transparent' },
        type: 'scatter',
        mode: 'lines',
        showlegend: false,
        hoverinfo: 'skip'
    };

    // Confidence band for generated
    const bandGenerated = {
        x: [...smoothMonths, ...smoothMonths.slice().reverse()],
        y: [...fitGenerated.yUpper, ...fitGenerated.yLower.slice().reverse()],
        fill: 'toself',
        fillcolor: 'rgba(231, 76, 60, 0.15)',
        line: { color: 'transparent' },
        type: 'scatter',
        mode: 'lines',
        showlegend: false,
        hoverinfo: 'skip'
    };

    // Smooth polynomial fit lines
    const lineAssisted = {
        x: smoothMonths,
        y: fitAssisted.ySmooth,
        name: 'AI-Generated or AI-Assisted',
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS.aiAssisted, width: 2.5 },
        hoverinfo: 'skip'
    };

    const lineGenerated = {
        x: smoothMonths,
        y: fitGenerated.ySmooth,
        name: 'Fully AI-Generated',
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS.aiGenerated, width: 2.5 },
        hoverinfo: 'skip'
    };

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 30, r: 30, b: 60, l: 65 },
        xaxis: {
            title: { text: '' },
            type: 'date',
            tickformat: '%b %Y',
            dtick: 'M3',
            gridcolor: '#f0f0f0',
            zeroline: false
        },
        yaxis: {
            title: { text: 'Share of Websites (%)', font: { size: 14 } },
            gridcolor: '#f0f0f0',
            zeroline: true,
            zerolinecolor: '#e0e0e0',
            range: [0, 42]
        },
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.85)',
            bordercolor: '#e0e0e0',
            borderwidth: 1,
            font: { size: 13 }
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        shapes: [{
            type: 'line',
            x0: '2022-11-30',
            x1: '2022-11-30',
            y0: 0,
            y1: 42,
            line: { color: '#999', width: 1.5, dash: 'dot' }
        }],
        annotations: [{
            x: '2022-11-30',
            y: 38,
            text: 'ChatGPT Launch',
            showarrow: false,
            font: { size: 12, color: '#666', family: MONO_FONT },
            xanchor: 'left',
            xshift: 8
        }],
        hovermode: 'x unified'
    };

    Plotly.newPlot('prevalence-plot',
        [bandAssisted, bandGenerated, scatterAssisted, scatterGenerated, lineAssisted, lineGenerated],
        layout, PLOTLY_CONFIG);
}

function plotHypothesisScatter(hypKey) {
    const hyp = SURVEY_DATA[hypKey];
    const points = generateScatterData(hyp.rho, hyp.scatterSeed, hyp.yMean, hyp.yStd);

    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);

    // Compute regression line
    const n = xs.length;
    const sumX = xs.reduce((a, b) => a + b, 0);
    const sumY = ys.reduce((a, b) => a + b, 0);
    const sumXY = xs.reduce((a, b, i) => a + b * ys[i], 0);
    const sumX2 = xs.reduce((a, b) => a + b * b, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const regX = [xMin - 0.01, xMax + 0.01];
    const regY = regX.map(x => slope * x + intercept);

    const dotColor = hyp.confirmed ? COLORS.confirmed : COLORS.scatter;

    const scatterTrace = {
        x: xs,
        y: ys,
        type: 'scatter',
        mode: 'markers',
        marker: {
            color: dotColor,
            size: 8,
            opacity: 0.7,
            line: { color: 'white', width: 1 }
        },
        hovertemplate: `AI Likelihood: %{x:.3f}<br>${hyp.yLabel}: %{y:.4f}<extra></extra>`,
        showlegend: false
    };

    const regTrace = {
        x: regX,
        y: regY,
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS.regression, width: 2, dash: 'dash' },
        showlegend: false,
        hoverinfo: 'skip'
    };

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 15, r: 15, b: 50, l: 60 },
        xaxis: {
            title: { text: 'AI Likelihood', font: { size: 11 } },
            gridcolor: '#f0f0f0',
            zeroline: false,
            tickfont: { size: 10 }
        },
        yaxis: {
            title: { text: hyp.yLabel, font: { size: 11 } },
            gridcolor: '#f0f0f0',
            zeroline: false,
            tickfont: { size: 10 }
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        annotations: [{
            x: xMax,
            y: regY[1],
            text: `ρ = ${hyp.rho}, p = ${hyp.p < 0.001 ? hyp.p.toExponential(1) : hyp.p}`,
            showarrow: false,
            font: { size: 11, color: hyp.confirmed ? COLORS.confirmed : '#666', family: MONO_FONT },
            xanchor: 'right',
            yanchor: 'bottom',
            yshift: 10
        }]
    };

    Plotly.newPlot(`scatter-${hypKey}`, [scatterTrace, regTrace], layout, PLOTLY_CONFIG);
}

// Dual-axis time series plot
function plotHypothesisTimeSeries(hypKey) {
    const hyp = SURVEY_DATA[hypKey];
    const ts = generateTimeSeriesData(hyp.rho, hyp.scatterSeed, hyp.yMean, hyp.yStd);

    const signalColor = hyp.confirmed ? COLORS.confirmed : COLORS.scatter;
    const aiColor = COLORS.aiAssisted;

    const traceSignal = {
        x: ts.months,
        y: ts.signal,
        name: hyp.yLabel,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: signalColor, width: 2.5, shape: 'spline' },
        marker: { color: signalColor, size: 4 },
        yaxis: 'y',
        hovertemplate: `%{x|%b %Y}<br>${hyp.yLabel}: %{y:.4f}<extra></extra>`
    };

    const traceAI = {
        x: ts.months,
        y: ts.aiLikelihood,
        name: 'AI Likelihood',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: aiColor, width: 2.5, shape: 'spline', dash: 'dash' },
        marker: { color: aiColor, size: 4 },
        yaxis: 'y2',
        hovertemplate: '%{x|%b %Y}<br>AI Likelihood: %{y:.3f}<extra></extra>'
    };

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 15, r: 60, b: 50, l: 60 },
        xaxis: {
            type: 'date',
            tickformat: '%b %Y',
            dtick: 'M6',
            gridcolor: '#f0f0f0',
            zeroline: false,
            tickfont: { size: 10 }
        },
        yaxis: {
            title: { text: hyp.yLabel, font: { size: 11, color: signalColor } },
            gridcolor: '#f0f0f0',
            zeroline: false,
            tickfont: { size: 10, color: signalColor },
            side: 'left'
        },
        yaxis2: {
            title: { text: 'AI Likelihood', font: { size: 11, color: aiColor } },
            overlaying: 'y',
            side: 'right',
            gridcolor: 'transparent',
            zeroline: false,
            tickfont: { size: 10, color: aiColor }
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.85)',
            bordercolor: '#e0e0e0',
            borderwidth: 1,
            font: { size: 10 }
        },
        shapes: [{
            type: 'line',
            x0: '2022-11-30',
            x1: '2022-11-30',
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: '#999', width: 1, dash: 'dot' }
        }],
        hovermode: 'x unified'
    };

    Plotly.newPlot(`timeseries-${hypKey}`, [traceSignal, traceAI], layout, PLOTLY_CONFIG);
}

function plotSurveyOverall(hypKey) {
    const hyp = SURVEY_DATA[hypKey];
    const d = hyp.overall;
    const total = d.SD + d.D + d.SoD + d.N + d.SoA + d.A + d.SA;

    const categories = ['SD', 'D', 'SoD', 'N', 'SoA', 'A', 'SA'];
    const fullNames = ['Strongly<br>Disagree', 'Disagree', 'Somewhat<br>Disagree', 'Neutral', 'Somewhat<br>Agree', 'Agree', 'Strongly<br>Agree'];
    const colors = categories.map(c => COLORS.likert[c]);
    const values = categories.map(c => Math.round(d[c] / total * 1000) / 10);

    const trace = {
        x: fullNames,
        y: values,
        type: 'bar',
        marker: {
            color: colors,
            line: { color: 'white', width: 1 }
        },
        hovertemplate: '%{x}: %{y:.1f}%<extra></extra>',
        showlegend: false
    };

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 8, r: 10, b: 60, l: 45 },
        xaxis: {
            tickfont: { size: 10 },
            fixedrange: true
        },
        yaxis: {
            title: { text: '%', font: { size: 12 } },
            gridcolor: '#f0f0f0',
            zeroline: false,
            fixedrange: true,
            range: [0, Math.max(...values) * 1.15]
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        bargap: 0.15
    };

    Plotly.newPlot(`survey-overall-${hypKey}`, [trace], layout, PLOTLY_CONFIG);
}

function plotSurveyByUsage(hypKey) {
    const hyp = SURVEY_DATA[hypKey];
    const usageGroups = ["Never", "Monthly", "Weekly", "Daily"];
    const categories = ['SD', 'D', 'SoD', 'N', 'SoA', 'A', 'SA'];
    const fullNames = {
        SD: 'Strongly Disagree',
        D: 'Disagree',
        SoD: 'Somewhat Disagree',
        N: 'Neutral',
        SoA: 'Somewhat Agree',
        A: 'Agree',
        SA: 'Strongly Agree'
    };

    const traces = categories.map(cat => {
        const vals = usageGroups.map(g => {
            const d = hyp.byUsage[g];
            const total = Object.values(d).reduce((a, b) => a + b, 0);
            return Math.round(d[cat] / total * 1000) / 10;
        });
        return {
            x: usageGroups,
            y: vals,
            name: fullNames[cat],
            type: 'bar',
            marker: { color: COLORS.likert[cat] },
            hovertemplate: `${fullNames[cat]}: %{y:.1f}%<extra></extra>`
        };
    });

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 8, r: 10, b: 45, l: 45 },
        barmode: 'stack',
        xaxis: {
            title: { text: 'AI Usage Frequency', font: { size: 11 } },
            tickfont: { size: 11 },
            fixedrange: true
        },
        yaxis: {
            title: { text: '%', font: { size: 12 } },
            gridcolor: '#f0f0f0',
            zeroline: false,
            fixedrange: true,
            range: [0, 105]
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        showlegend: false,
        bargap: 0.2
    };

    Plotly.newPlot(`survey-usage-${hypKey}`, traces, layout, PLOTLY_CONFIG);
}

function plotSurveyByView(hypKey) {
    const hyp = SURVEY_DATA[hypKey];
    const viewGroups = ["Negative", "Neutral", "Positive"];
    const categories = ['SD', 'D', 'SoD', 'N', 'SoA', 'A', 'SA'];
    const fullNames = {
        SD: 'Strongly Disagree',
        D: 'Disagree',
        SoD: 'Somewhat Disagree',
        N: 'Neutral',
        SoA: 'Somewhat Agree',
        A: 'Agree',
        SA: 'Strongly Agree'
    };

    const traces = categories.map(cat => {
        const vals = viewGroups.map(g => {
            const d = hyp.byView[g];
            const total = Object.values(d).reduce((a, b) => a + b, 0);
            return Math.round(d[cat] / total * 1000) / 10;
        });
        return {
            x: viewGroups,
            y: vals,
            name: fullNames[cat],
            type: 'bar',
            marker: { color: COLORS.likert[cat] },
            hovertemplate: `${fullNames[cat]}: %{y:.1f}%<extra></extra>`
        };
    });

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 8, r: 10, b: 45, l: 45 },
        barmode: 'stack',
        xaxis: {
            title: { text: 'View of AI Impact', font: { size: 11 } },
            tickfont: { size: 11 },
            fixedrange: true
        },
        yaxis: {
            title: { text: '%', font: { size: 12 } },
            gridcolor: '#f0f0f0',
            zeroline: false,
            fixedrange: true,
            range: [0, 105]
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        showlegend: false,
        bargap: 0.2
    };

    Plotly.newPlot(`survey-view-${hypKey}`, traces, layout, PLOTLY_CONFIG);
}

// Summary comparison chart
function plotHypothesisSummary() {
    const hypotheses = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
    const names = hypotheses.map(h => SURVEY_DATA[h].name);
    const rhos = hypotheses.map(h => SURVEY_DATA[h].rho);
    const confirmed = hypotheses.map(h => SURVEY_DATA[h].confirmed);

    // Agreement percentages from survey
    const agreePcts = hypotheses.map(h => {
        const d = SURVEY_DATA[h].overall;
        const total = d.SD + d.D + d.SoD + d.N + d.SoA + d.A + d.SA;
        return Math.round((d.SoA + d.A + d.SA) / total * 1000) / 10;
    });

    // Correlation bars
    const traceRho = {
        y: names.slice().reverse(),
        x: rhos.slice().reverse(),
        type: 'bar',
        orientation: 'h',
        name: 'Correlation (ρ)',
        marker: {
            color: confirmed.slice().reverse().map(c => c ? COLORS.confirmed : COLORS.notConfirmed),
            line: { color: 'white', width: 1 }
        },
        hovertemplate: '%{y}<br>ρ = %{x}<extra></extra>',
        xaxis: 'x',
        yaxis: 'y'
    };

    // Agreement bars
    const traceAgree = {
        y: names.slice().reverse(),
        x: agreePcts.slice().reverse(),
        type: 'bar',
        orientation: 'h',
        name: 'Public Agreement (%)',
        marker: {
            color: '#3498db',
            opacity: 0.7,
            line: { color: 'white', width: 1 }
        },
        hovertemplate: '%{y}<br>Agreement: %{x:.1f}%<extra></extra>',
        xaxis: 'x2',
        yaxis: 'y2'
    };

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        grid: { rows: 1, columns: 2, pattern: 'independent' },
        margin: { t: 30, r: 30, b: 50, l: 160 },
        xaxis: {
            title: { text: 'Correlation (ρ)', font: { size: 13 } },
            domain: [0, 0.45],
            zeroline: true,
            zerolinecolor: '#ccc',
            gridcolor: '#f0f0f0',
            range: [-0.4, 0.7]
        },
        yaxis: {
            anchor: 'x',
            tickfont: { size: 12 }
        },
        xaxis2: {
            title: { text: 'Public Agreement (%)', font: { size: 13 } },
            domain: [0.55, 1],
            gridcolor: '#f0f0f0',
            range: [0, 100]
        },
        yaxis2: {
            anchor: 'x2',
            showticklabels: false
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        showlegend: false,
        annotations: [
            { text: '<b>Statistical Evidence</b>', x: 0.22, y: 1.06, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 13 } },
            { text: '<b>Public Belief</b>', x: 0.78, y: 1.06, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 13 } }
        ]
    };

    Plotly.newPlot('summary-plot', [traceRho, traceAgree], layout, PLOTLY_CONFIG);
}

// Shared legend
function plotSurveyLegend() {
    const categories = ['SD', 'D', 'SoD', 'N', 'SoA', 'A', 'SA'];
    const fullNames = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Neutral', 'Somewhat Agree', 'Agree', 'Strongly Agree'];

    const container = document.getElementById('survey-legend');
    if (!container) return;

    container.innerHTML = categories.map((cat, i) =>
        `<span style="display:inline-flex;align-items:center;margin-right:1em;margin-bottom:0.3em;font-size:12px;font-family:${FONT_FAMILY};color:#555;">` +
        `<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:${COLORS.likert[cat]};margin-right:4px;"></span>` +
        `${fullNames[i]}</span>`
    ).join('');
}

// ============================================================
// INIT
// ============================================================

document.addEventListener('DOMContentLoaded', function() {
    plotPrevalence();
    plotHypothesisSummary();

    const hypotheses = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
    hypotheses.forEach(h => {
        plotHypothesisScatter(h);
        plotHypothesisTimeSeries(h);
        plotSurveyOverall(h);
        plotSurveyByUsage(h);
        plotSurveyByView(h);
    });

    plotSurveyLegend();

    // Resize all plots on window resize
    window.addEventListener('resize', function() {
        document.querySelectorAll('.js-plotly-plot').forEach(el => {
            Plotly.Plots.resize(el);
        });
    });
});
