// ============================================================
// CSV LOADING
// ============================================================

async function loadCSV(url) {
    const resp = await fetch(url);
    const text = await resp.text();
    const lines = text.trim().replace(/\r/g, '').split('\n');
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
        const vals = line.split(',');
        const obj = {};
        headers.forEach((h, i) => {
            const v = vals[i];
            obj[h] = v === '' || v === undefined ? null : Number(v);
        });
        return obj;
    });
}

// String-preserving CSV loader (used for survey CSVs whose columns mix
// strings and numbers). Numeric columns can be coerced by callers.
async function loadCSVRaw(url) {
    const resp = await fetch(url);
    const text = await resp.text();
    const lines = text.trim().replace(/\r/g, '').split('\n');
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
        const vals = line.split(',');
        const obj = {};
        headers.forEach((h, i) => { obj[h] = vals[i]; });
        return obj;
    });
}

// ============================================================
// POLYNOMIAL FITTING (used only for prevalence plot)
// ============================================================

function polyFit(xArr, yArr, degree, nSmooth) {
    const n = xArr.length;
    degree = Math.min(degree, n - 1);

    const xMin = Math.min(...xArr);
    const xMax = Math.max(...xArr);
    const xRange = xMax - xMin || 1;
    const xNorm = xArr.map(x => (x - xMin) / xRange * 2 - 1);

    const coeffs = polyFitCoeffs(xNorm, yArr, degree);

    const xSmooth = [];
    for (let i = 0; i < nSmooth; i++) {
        xSmooth.push(xMin + (xMax - xMin) * i / (nSmooth - 1));
    }
    const xSmoothNorm = xSmooth.map(x => (x - xMin) / xRange * 2 - 1);
    const ySmooth = xSmoothNorm.map(x => polyEval(coeffs, x));

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

function polyFitCoeffs(x, y, degree) {
    const n = x.length;
    const m = degree + 1;
    const V = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < m; j++) {
            row.push(Math.pow(x[i], degree - j));
        }
        V.push(row);
    }
    const VtV = [];
    for (let i = 0; i < m; i++) {
        VtV.push([]);
        for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) sum += V[k][i] * V[k][j];
            VtV[i].push(sum);
        }
    }
    const Vty = [];
    for (let i = 0; i < m; i++) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += V[k][i] * y[k];
        Vty.push(sum);
    }
    return gaussianSolve(VtV, Vty);
}

function gaussianSolve(A, b) {
    const n = A.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
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

// Linear regression returning slope, intercept
function linearFit(xs, ys) {
    const n = xs.length;
    const sumX = xs.reduce((a, b) => a + b, 0);
    const sumY = ys.reduce((a, b) => a + b, 0);
    const sumXY = xs.reduce((a, b, i) => a + b * ys[i], 0);
    const sumX2 = xs.reduce((a, b) => a + b * b, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    return { slope, intercept };
}

// ============================================================
// SURVEY DATA — quantitative metadata only.
// `overall`, `byUsage`, `byView`, and `n` are populated at load time from
// data/survey_hypotheses.csv (see populateSurveyData below).
// rho/p/confirmed and the signal/yLabel fields describe the quantitative
// internet-archive analysis, not the survey, and stay hardcoded.
// ============================================================

const SURVEY_DATA = {
    h1: {
        name: "Semantic Contraction",
        rho: 0.47, p: 0.004, confirmed: true,
        signalKey: 'h1_cosine_similarity',
        yLabel: "Avg. Pairwise Cosine Similarity",
        yMin: 0
    },
    h2: {
        name: "Truth Decay",
        rho: -0.19, p: 0.27, confirmed: false,
        signalKey: 'h2_error_rate',
        yLabel: "Factual Error Rate"
    },
    h3: {
        name: "Positivity Shift",
        rho: 0.56, p: 0.0003, confirmed: true,
        signalKey: 'h3_positive_rate',
        yLabel: "Rate of Positive Documents",
        yMin: 0
    },
    h4: {
        name: "Epistemic Islands",
        rho: -0.12, p: 0.48, confirmed: false,
        signalKey: 'h4_link_density',
        yLabel: "Outbound Link Density (per 1k words)"
    },
    h5: {
        name: "Entropy Dilution",
        rho: -0.02, p: 0.89, confirmed: false,
        signalKey: 'h5_compression_ratio',
        yLabel: "Gzip Compression Ratio"
    },
    h6: {
        name: "Stylistic Monoculture",
        rho: 0.24, p: 0.17, confirmed: false,
        signalKey: 'h6_jaccard_similarity',
        yLabel: "Avg. Pairwise Jaccard Similarity (3-gram)"
    }
};

// Populated from data/survey_overall.csv at load time.
const OVERALL_SURVEY = { ai_usage: {}, ai_view: {} };

// Likert columns in canonical order — must match the survey CSV headers.
const LIKERT_KEYS = ['SD', 'D', 'SoD', 'N', 'SoA', 'A', 'SA'];

function populateSurveyData(rows) {
    rows.forEach(r => {
        const hyp = SURVEY_DATA[r.hypothesis];
        if (!hyp) return;
        const counts = {};
        LIKERT_KEYS.forEach(k => { counts[k] = Number(r[k]) || 0; });
        if (r.split === 'overall') {
            hyp.overall = counts;
            hyp.n = Number(r.n) || 0;
        } else if (r.split === 'by_usage') {
            hyp.byUsage = hyp.byUsage || {};
            hyp.byUsage[r.group] = counts;
        } else if (r.split === 'by_view') {
            hyp.byView = hyp.byView || {};
            hyp.byView[r.group] = counts;
        }
    });
}

function populateOverallSurvey(rows) {
    rows.forEach(r => {
        if (r.dimension === 'ai_usage' || r.dimension === 'ai_view') {
            OVERALL_SURVEY[r.dimension][r.bucket] = {
                count: Number(r.count) || 0,
                pct: Number(r.pct) || 0,
            };
        }
    });
}

// ============================================================
// PLOTLY CONFIG & STYLE
// ============================================================

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

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
        SD: '#c0392b', D: '#e74c3c', SoD: '#f39c9c',
        N: '#bdc3c7',
        SoA: '#85c1e9', A: '#3498db', SA: '#1a5276'
    }
};

// ============================================================
// PLOT FUNCTIONS
// ============================================================

function plotPrevalence(csvRows) {
    const months = csvRows.map(r => r.month + '-15');
    const aiGenerated = csvRows.map(r => r.ai_generated_pct);
    const aiCombined = csvRows.map(r => r.ai_combined_pct);
    const xIndices = csvRows.map((_, i) => i);

    const monthLabels = csvRows.map(r => {
        const [y, m] = r.month.split('-');
        const names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        return names[parseInt(m) - 1] + ' ' + y;
    });

    const degree = 6;
    const nSmooth = 300;
    const fitCombined = polyFit(xIndices, aiCombined, degree, nSmooth);
    const fitGenerated = polyFit(xIndices, aiGenerated, degree, nSmooth);

    const smoothMonths = fitCombined.xSmooth.map(xi => {
        const base = Math.floor(xi);
        const frac = xi - base;
        if (base >= 0 && base < months.length - 1) {
            const d0 = new Date(months[base]);
            const d1 = new Date(months[Math.min(base + 1, months.length - 1)]);
            return new Date(d0.getTime() + frac * (d1.getTime() - d0.getTime())).toISOString().slice(0, 10);
        }
        return base >= months.length - 1 ? months[months.length - 1] : months[0];
    });

    const traces = [
        // Confidence bands (back)
        {
            x: [...smoothMonths, ...smoothMonths.slice().reverse()],
            y: [...fitCombined.yUpper, ...fitCombined.yLower.slice().reverse()],
            fill: 'toself', fillcolor: 'rgba(142, 68, 173, 0.15)',
            line: { color: 'transparent' }, type: 'scatter', mode: 'lines',
            showlegend: false, hoverinfo: 'skip'
        },
        {
            x: [...smoothMonths, ...smoothMonths.slice().reverse()],
            y: [...fitGenerated.yUpper, ...fitGenerated.yLower.slice().reverse()],
            fill: 'toself', fillcolor: 'rgba(231, 76, 60, 0.15)',
            line: { color: 'transparent' }, type: 'scatter', mode: 'lines',
            showlegend: false, hoverinfo: 'skip'
        },
        // Red (generated) — drawn first so purple goes on top
        {
            x: months, y: aiGenerated, name: '', type: 'scatter', mode: 'markers',
            marker: { color: COLORS.aiGenerated, size: 6, opacity: 0.4, symbol: 'circle' },
            hovertemplate: '<b>%{text}</b><br>Fully AI-Generated: %{y:.1f}%<extra></extra>',
            text: monthLabels, showlegend: false
        },
        {
            x: smoothMonths, y: fitGenerated.ySmooth, name: 'Fully AI-Generated',
            type: 'scatter', mode: 'lines',
            line: { color: COLORS.aiGenerated, width: 2.5 }, hoverinfo: 'skip'
        },
        // Purple (combined) — drawn last so it's on top
        {
            x: months, y: aiCombined, name: '', type: 'scatter', mode: 'markers',
            marker: { color: COLORS.aiAssisted, size: 7, opacity: 0.5, symbol: 'square' },
            hovertemplate: '<b>%{text}</b><br>AI-Gen. or Assisted: %{y:.1f}%<extra></extra>',
            text: monthLabels, showlegend: false
        },
        {
            x: smoothMonths, y: fitCombined.ySmooth, name: 'AI-Generated or AI-Assisted',
            type: 'scatter', mode: 'lines',
            line: { color: COLORS.aiAssisted, width: 2.5 }, hoverinfo: 'skip'
        }
    ];

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 30, r: 30, b: 60, l: 65 },
        xaxis: {
            type: 'date', tickformat: '%b %Y', dtick: 'M3',
            gridcolor: '#f0f0f0', zeroline: false,
            range: ['2022-06-01', '2025-06-01']
        },
        yaxis: {
            title: { text: 'Share of Websites (%)', font: { size: 14 } },
            gridcolor: '#f0f0f0', zeroline: true, zerolinecolor: '#e0e0e0',
            range: [0, 42]
        },
        legend: {
            x: 0.02, y: 0.98,
            bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#e0e0e0', borderwidth: 1,
            font: { size: 13 }
        },
        plot_bgcolor: 'white', paper_bgcolor: 'white',
        shapes: [{
            type: 'line', x0: '2022-11-30', x1: '2022-11-30', y0: 0, y1: 42,
            line: { color: '#999', width: 1.5, dash: 'dot' }
        }],
        annotations: [{
            x: '2022-11-30', y: 21, text: 'ChatGPT Launch', showarrow: false,
            font: { size: 12, color: '#666', family: MONO_FONT },
            xanchor: 'left', xshift: 8
        }],
        hovermode: 'x unified'
    };

    Plotly.newPlot('prevalence-plot', traces, layout, PLOTLY_CONFIG);
}

function plotHypothesisScatter(hypKey, hypRows) {
    const hyp = SURVEY_DATA[hypKey];
    const key = hyp.signalKey;

    const valid = hypRows.filter(r => r[key] !== null && r.ai_likelihood !== null);
    const xs = valid.map(r => r.ai_likelihood);
    const ys = valid.map(r => r[key]);

    const { slope, intercept } = linearFit(xs, ys);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const pad = (xMax - xMin) * 0.05;
    const regX = [xMin - pad, xMax + pad];
    const regY = regX.map(x => slope * x + intercept);

    const dotColor = hyp.confirmed ? COLORS.confirmed : COLORS.scatter;

    const scatterTrace = {
        x: xs, y: ys, type: 'scatter', mode: 'markers',
        marker: { color: dotColor, size: 8, opacity: 0.7, line: { color: 'white', width: 1 } },
        hovertemplate: `AI Likelihood: %{x:.3f}<br>${hyp.yLabel}: %{y:.4f}<extra></extra>`,
        showlegend: false
    };

    const regTrace = {
        x: regX, y: regY, type: 'scatter', mode: 'lines',
        line: { color: COLORS.regression, width: 2, dash: 'dash' },
        showlegend: false, hoverinfo: 'skip'
    };

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 15, r: 15, b: 50, l: 60 },
        xaxis: {
            title: { text: 'AI Likelihood', font: { size: 11 } },
            gridcolor: '#f0f0f0', zeroline: false, tickfont: { size: 10 }
        },
        yaxis: {
            title: { text: hyp.yLabel, font: { size: 11 } },
            gridcolor: '#f0f0f0', zeroline: false, tickfont: { size: 10 }
        },
        plot_bgcolor: 'white', paper_bgcolor: 'white',
        annotations: [{
            x: xMax, y: regY[1],
            text: `ρ = ${hyp.rho}, p = ${hyp.p < 0.001 ? hyp.p.toExponential(1) : hyp.p}`,
            showarrow: false,
            font: { size: 11, color: hyp.confirmed ? COLORS.confirmed : '#666', family: MONO_FONT },
            xanchor: 'right', yanchor: 'bottom', yshift: 10
        }]
    };

    Plotly.newPlot(`scatter-${hypKey}`, [scatterTrace, regTrace], layout, PLOTLY_CONFIG);
}

function plotHypothesisTimeSeries(hypKey, hypRows) {
    const hyp = SURVEY_DATA[hypKey];
    const key = hyp.signalKey;

    const valid = hypRows.filter(r => r[key] !== null && r.ai_likelihood !== null);
    const months = valid.map(r => r.month + '-15');
    const signal = valid.map(r => r[key]);
    const aiLikelihood = valid.map(r => r.ai_likelihood);

    const signalColor = hyp.confirmed ? COLORS.confirmed : COLORS.scatter;
    const aiColor = COLORS.aiAssisted;

    // Use actual temporal positions (months since epoch) for linear fitting
    const xNums = valid.map(r => {
        const [y, m] = r.month.split('-');
        return parseInt(y) * 12 + parseInt(m);
    });

    const fitSignal = linearFit(xNums, signal);
    const fitAI = linearFit(xNums, aiLikelihood);
    const trendSignalY = xNums.map(x => fitSignal.slope * x + fitSignal.intercept);
    const trendAIY = xNums.map(x => fitAI.slope * x + fitAI.intercept);

    const traceSignal = {
        x: months, y: signal, name: hyp.yLabel,
        type: 'scatter', mode: 'markers',
        marker: { color: signalColor, size: 5, opacity: 0.5 },
        yaxis: 'y',
        hovertemplate: `%{x|%b %Y}<br>${hyp.yLabel}: %{y:.4f}<extra></extra>`
    };

    const traceSignalTrend = {
        x: months, y: trendSignalY, name: hyp.yLabel + ' (trend)',
        type: 'scatter', mode: 'lines',
        line: { color: signalColor, width: 2.5 },
        yaxis: 'y', showlegend: false, hoverinfo: 'skip'
    };

    const traceAI = {
        x: months, y: aiLikelihood, name: 'AI Likelihood',
        type: 'scatter', mode: 'markers',
        marker: { color: aiColor, size: 5, opacity: 0.5 },
        yaxis: 'y2',
        hovertemplate: '%{x|%b %Y}<br>AI Likelihood: %{y:.3f}<extra></extra>'
    };

    const traceAITrend = {
        x: months, y: trendAIY, name: 'AI Likelihood (trend)',
        type: 'scatter', mode: 'lines',
        line: { color: aiColor, width: 2.5, dash: 'dash' },
        yaxis: 'y2', showlegend: false, hoverinfo: 'skip'
    };

    const layout = {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 15, r: 60, b: 50, l: 60 },
        xaxis: {
            type: 'date', tickformat: '%b %Y', dtick: 'M6',
            gridcolor: '#f0f0f0', zeroline: false, tickfont: { size: 10 }
        },
        yaxis: {
            title: { text: hyp.yLabel, font: { size: 11, color: signalColor } },
            gridcolor: '#f0f0f0', zeroline: false,
            tickfont: { size: 10, color: signalColor }, side: 'left',
            ...(hyp.yMin !== undefined && { range: [hyp.yMin, Math.max(...signal) * 1.1] })
        },
        yaxis2: {
            title: { text: 'AI Likelihood', font: { size: 11, color: aiColor } },
            overlaying: 'y', side: 'right', gridcolor: 'transparent',
            zeroline: false, tickfont: { size: 10, color: aiColor },
            range: [0, 1]
        },
        plot_bgcolor: 'white', paper_bgcolor: 'white',
        legend: {
            x: 0.02, y: 0.98,
            bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#e0e0e0', borderwidth: 1,
            font: { size: 10 }
        },
        shapes: [{
            type: 'line', x0: '2022-11-30', x1: '2022-11-30', y0: 0, y1: 1,
            yref: 'paper', line: { color: '#999', width: 1, dash: 'dot' }
        }],
        hovermode: 'x unified'
    };

    Plotly.newPlot(`timeseries-${hypKey}`, [traceSignal, traceSignalTrend, traceAI, traceAITrend], layout, PLOTLY_CONFIG);
}

function plotSurveyOverall(hypKey) {
    const hyp = SURVEY_DATA[hypKey];
    const d = hyp.overall;
    const total = d.SD + d.D + d.SoD + d.N + d.SoA + d.A + d.SA;
    const categories = ['SD', 'D', 'SoD', 'N', 'SoA', 'A', 'SA'];
    const fullNames = ['Strongly<br>Disagree', 'Disagree', 'Somewhat<br>Disagree', 'Neutral', 'Somewhat<br>Agree', 'Agree', 'Strongly<br>Agree'];
    const colors = categories.map(c => COLORS.likert[c]);
    const values = categories.map(c => Math.round(d[c] / total * 1000) / 10);

    Plotly.newPlot(`survey-overall-${hypKey}`, [{
        x: fullNames, y: values, type: 'bar',
        marker: { color: colors, line: { color: 'white', width: 1 } },
        hovertemplate: '%{x}: %{y:.1f}%<extra></extra>', showlegend: false
    }], {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 8, r: 10, b: 60, l: 45 },
        xaxis: { tickfont: { size: 10 }, fixedrange: true },
        yaxis: {
            title: { text: '%', font: { size: 12 } }, gridcolor: '#f0f0f0',
            zeroline: false, fixedrange: true, range: [0, Math.max(...values) * 1.15]
        },
        plot_bgcolor: 'white', paper_bgcolor: 'white', bargap: 0.15
    }, PLOTLY_CONFIG);
}

function plotSurveyByUsage(hypKey) {
    const hyp = SURVEY_DATA[hypKey];
    const usageGroups = ["Never", "Monthly", "Weekly", "Daily"];
    const categories = ['SD', 'D', 'SoD', 'N', 'SoA', 'A', 'SA'];
    const fullNames = { SD: 'Strongly Disagree', D: 'Disagree', SoD: 'Somewhat Disagree', N: 'Neutral', SoA: 'Somewhat Agree', A: 'Agree', SA: 'Strongly Agree' };

    const traces = categories.map(cat => ({
        x: usageGroups,
        y: usageGroups.map(g => {
            const d = hyp.byUsage[g];
            const total = Object.values(d).reduce((a, b) => a + b, 0);
            return Math.round(d[cat] / total * 1000) / 10;
        }),
        name: fullNames[cat], type: 'bar',
        marker: { color: COLORS.likert[cat] },
        hovertemplate: `${fullNames[cat]}: %{y:.1f}%<extra></extra>`
    }));

    Plotly.newPlot(`survey-usage-${hypKey}`, traces, {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 8, r: 10, b: 45, l: 45 },
        barmode: 'stack',
        xaxis: { tickfont: { size: 11 }, fixedrange: true },
        yaxis: { title: { text: '%', font: { size: 12 } }, gridcolor: '#f0f0f0', zeroline: false, fixedrange: true, range: [0, 105] },
        plot_bgcolor: 'white', paper_bgcolor: 'white', showlegend: false, bargap: 0.2
    }, PLOTLY_CONFIG);
}

function plotSurveyByView(hypKey) {
    const hyp = SURVEY_DATA[hypKey];
    const viewGroups = ["Negative", "Neutral", "Positive"];
    const categories = ['SD', 'D', 'SoD', 'N', 'SoA', 'A', 'SA'];
    const fullNames = { SD: 'Strongly Disagree', D: 'Disagree', SoD: 'Somewhat Disagree', N: 'Neutral', SoA: 'Somewhat Agree', A: 'Agree', SA: 'Strongly Agree' };

    const traces = categories.map(cat => ({
        x: viewGroups,
        y: viewGroups.map(g => {
            const d = hyp.byView[g];
            const total = Object.values(d).reduce((a, b) => a + b, 0);
            return Math.round(d[cat] / total * 1000) / 10;
        }),
        name: fullNames[cat], type: 'bar',
        marker: { color: COLORS.likert[cat] },
        hovertemplate: `${fullNames[cat]}: %{y:.1f}%<extra></extra>`
    }));

    Plotly.newPlot(`survey-view-${hypKey}`, traces, {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 8, r: 10, b: 45, l: 45 },
        barmode: 'stack',
        xaxis: { tickfont: { size: 11 }, fixedrange: true },
        yaxis: { title: { text: '%', font: { size: 12 } }, gridcolor: '#f0f0f0', zeroline: false, fixedrange: true, range: [0, 105] },
        plot_bgcolor: 'white', paper_bgcolor: 'white', showlegend: false, bargap: 0.2
    }, PLOTLY_CONFIG);
}

function plotHypothesisSummary() {
    const hypotheses = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
    const names = hypotheses.map(h => SURVEY_DATA[h].name);
    const rhos = hypotheses.map(h => SURVEY_DATA[h].rho);
    const confirmed = hypotheses.map(h => SURVEY_DATA[h].confirmed);

    const agreePcts = hypotheses.map(h => {
        const d = SURVEY_DATA[h].overall;
        const total = d.SD + d.D + d.SoD + d.N + d.SoA + d.A + d.SA;
        return Math.round((d.SoA + d.A + d.SA) / total * 1000) / 10;
    });

    Plotly.newPlot('summary-plot', [
        {
            y: names.slice().reverse(), x: rhos.slice().reverse(),
            type: 'bar', orientation: 'h', name: 'Correlation (ρ)',
            marker: { color: confirmed.slice().reverse().map(c => c ? COLORS.confirmed : COLORS.notConfirmed), line: { color: 'white', width: 1 } },
            hovertemplate: '%{y}<br>ρ = %{x}<extra></extra>', xaxis: 'x', yaxis: 'y'
        },
        {
            y: names.slice().reverse(), x: agreePcts.slice().reverse(),
            type: 'bar', orientation: 'h', name: 'Public Agreement (%)',
            marker: { color: '#3498db', opacity: 0.7, line: { color: 'white', width: 1 } },
            hovertemplate: '%{y}<br>Agreement: %{x:.1f}%<extra></extra>', xaxis: 'x2', yaxis: 'y2'
        }
    ], {
        font: { family: FONT_FAMILY, color: '#353535' },
        grid: { rows: 1, columns: 2, pattern: 'independent' },
        margin: { t: 30, r: 30, b: 50, l: 160 },
        xaxis: { title: { text: 'Correlation (ρ)', font: { size: 13 } }, domain: [0, 0.45], zeroline: true, zerolinecolor: '#ccc', gridcolor: '#f0f0f0', range: [-0.4, 0.7] },
        yaxis: { anchor: 'x', tickfont: { size: 12 } },
        xaxis2: { title: { text: 'Public Agreement (%)', font: { size: 13 } }, domain: [0.55, 1], gridcolor: '#f0f0f0', range: [0, 100] },
        yaxis2: { anchor: 'x2', showticklabels: false },
        plot_bgcolor: 'white', paper_bgcolor: 'white', showlegend: false,
        annotations: [
            { text: '<b>Statistical Evidence</b>', x: 0.22, y: 1.06, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 13 } },
            { text: '<b>Public Belief</b>', x: 0.78, y: 1.06, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 13 } }
        ]
    }, PLOTLY_CONFIG);
}

function plotSurveyLegends() {
    const categories = ['SD', 'D', 'SoD', 'N', 'SoA', 'A', 'SA'];
    const fullNames = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Neutral', 'Somewhat Agree', 'Agree', 'Strongly Agree'];
    const html = categories.map((cat, i) =>
        `<span style="display:inline-flex;align-items:center;margin-right:1em;margin-bottom:0.3em;font-size:12px;font-family:${FONT_FAMILY};color:#555;">` +
        `<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:${COLORS.likert[cat]};margin-right:4px;"></span>` +
        `${fullNames[i]}</span>`
    ).join('');
    document.querySelectorAll('.survey-legend-inline').forEach(el => { el.innerHTML = html; });
}

function plotOverallSurveyDistributions() {
    // Usage distribution — read directly from data/survey_overall.csv
    const usageGroups = ["Never", "Monthly", "Weekly", "Daily"];
    const usagePcts = usageGroups.map(g =>
        Math.round((OVERALL_SURVEY.ai_usage[g]?.pct ?? 0) * 10) / 10
    );

    Plotly.newPlot('survey-usage-overall', [{
        x: usageGroups, y: usagePcts, type: 'bar',
        marker: { color: '#3498db', line: { color: 'white', width: 1 } },
        hovertemplate: '%{x}: %{y:.1f}%<extra></extra>', showlegend: false,
        text: usagePcts.map(v => v.toFixed(1) + '%'), textposition: 'outside'
    }], {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 20, r: 10, b: 40, l: 45 },
        xaxis: { tickfont: { size: 12 }, fixedrange: true },
        yaxis: { title: { text: '%', font: { size: 12 } }, gridcolor: '#f0f0f0', zeroline: false, fixedrange: true, range: [0, Math.max(...usagePcts) * 1.2] },
        plot_bgcolor: 'white', paper_bgcolor: 'white', bargap: 0.25
    }, PLOTLY_CONFIG);

    // View distribution — read directly from data/survey_overall.csv
    const viewGroups = ["Negative", "Neutral", "Positive"];
    const viewPcts = viewGroups.map(g =>
        Math.round((OVERALL_SURVEY.ai_view[g]?.pct ?? 0) * 10) / 10
    );

    Plotly.newPlot('survey-view-overall', [{
        x: viewGroups, y: viewPcts, type: 'bar',
        marker: { color: '#8e44ad', line: { color: 'white', width: 1 } },
        hovertemplate: '%{x}: %{y:.1f}%<extra></extra>', showlegend: false,
        text: viewPcts.map(v => v.toFixed(1) + '%'), textposition: 'outside'
    }], {
        font: { family: FONT_FAMILY, color: '#353535' },
        margin: { t: 20, r: 10, b: 40, l: 45 },
        xaxis: { tickfont: { size: 12 }, fixedrange: true },
        yaxis: { title: { text: '%', font: { size: 12 } }, gridcolor: '#f0f0f0', zeroline: false, fixedrange: true, range: [0, Math.max(...viewPcts) * 1.2] },
        plot_bgcolor: 'white', paper_bgcolor: 'white', bargap: 0.25
    }, PLOTLY_CONFIG);
}

// ============================================================
// INIT — load real data, then render all plots
// ============================================================

document.addEventListener('DOMContentLoaded', async function() {
    const [prevalenceData, hypothesesData, surveyOverallRows, surveyHypRows] = await Promise.all([
        loadCSV('data/prevalence.csv'),
        loadCSV('data/hypotheses.csv'),
        loadCSVRaw('data/survey_overall.csv'),
        loadCSVRaw('data/survey_hypotheses.csv')
    ]);

    populateOverallSurvey(surveyOverallRows);
    populateSurveyData(surveyHypRows);

    // Attach month strings to hypothesis rows (CSV month column is parsed as a number by our loader)
    const hypothesesRaw = await fetch('data/hypotheses.csv').then(r => r.text());
    const hypLines = hypothesesRaw.trim().split('\n');
    const hypHeaders = hypLines[0].split(',');
    const monthIdx = hypHeaders.indexOf('month');
    hypLines.slice(1).forEach((line, i) => {
        if (hypothesesData[i]) {
            hypothesesData[i].month = line.split(',')[monthIdx];
        }
    });

    // Same for prevalence
    const prevRaw = await fetch('data/prevalence.csv').then(r => r.text());
    const prevLines = prevRaw.trim().split('\n');
    const prevHeaders = prevLines[0].split(',');
    const prevMonthIdx = prevHeaders.indexOf('month');
    prevLines.slice(1).forEach((line, i) => {
        if (prevalenceData[i]) {
            prevalenceData[i].month = line.split(',')[prevMonthIdx];
        }
    });

    plotPrevalence(prevalenceData);
    plotOverallSurveyDistributions();
    plotHypothesisSummary();

    const hypotheses = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
    hypotheses.forEach(h => {
        plotHypothesisScatter(h, hypothesesData);
        plotHypothesisTimeSeries(h, hypothesesData);
        plotSurveyOverall(h);
        plotSurveyByUsage(h);
        plotSurveyByView(h);
    });

    plotSurveyLegends();

    window.addEventListener('resize', function() {
        document.querySelectorAll('.js-plotly-plot').forEach(el => {
            Plotly.Plots.resize(el);
        });
    });
});
