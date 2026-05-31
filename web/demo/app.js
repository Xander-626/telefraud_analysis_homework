const offlineSamples = [
  {
    sample_id: "normal_example",
    title: "正常通话样例",
    expected_label: "normal",
    description: "普通快递/生活通知类通话，缺少转账和威胁话术。",
    audio_url: "/api/audio/normal_example"
  },
  {
    sample_id: "fraud_example_1",
    title: "电诈通话样例 1",
    expected_label: "fraud",
    description: "冒充官方机构，要求受害者配合资金核验。",
    audio_url: "/api/audio/fraud_example_1"
  },
  {
    sample_id: "fraud_example_2",
    title: "电诈通话样例 2",
    expected_label: "fraud",
    description: "制造紧迫感并索要验证码，符合高风险诈骗特征。",
    audio_url: "/api/audio/fraud_example_2"
  }
];

const offlinePredictions = {
  normal_example: {
    sample_id: "normal_example",
    prediction: "normal",
    fraud_probability: 0.041,
    risk_level: "low",
    asr_text: "您好，这里是快递员，您的包裹已经放到门口，方便时请查收。",
    evidence: ["生活通知语境", "无转账要求", "无验证码索取"],
    model: "Whisper-small ASR + Chinese RoBERTa + MLP Fusion Classifier"
  },
  fraud_example_1: {
    sample_id: "fraud_example_1",
    prediction: "fraud",
    fraud_probability: 0.982,
    risk_level: "high",
    asr_text: "你的银行卡涉嫌异常交易，请马上把资金转入安全账户配合核查。",
    evidence: ["银行卡涉嫌异常", "转入安全账户", "紧急核查话术"],
    model: "Whisper-small ASR + Chinese RoBERTa + MLP Fusion Classifier"
  },
  fraud_example_2: {
    sample_id: "fraud_example_2",
    prediction: "fraud",
    fraud_probability: 0.964,
    risk_level: "high",
    asr_text: "系统显示你的账户存在风险，请提供短信验证码完成解冻。",
    evidence: ["账户风险恐吓", "索要短信验证码", "解冻诱导"],
    model: "Whisper-small ASR + Chinese RoBERTa + MLP Fusion Classifier"
  }
};

const sampleList = document.querySelector("#sample-list");
const resultPanel = document.querySelector("#result-panel");
const textInput = document.querySelector("#text-input");
const textDetect = document.querySelector("#text-detect");
const uploadDetect = document.querySelector("#upload-detect");
const uploadName = document.querySelector("#upload-name");
const audioFile = document.querySelector("#audio-file");

let selectedSampleId = "";

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
});

textDetect.addEventListener("click", detectText);
uploadDetect.addEventListener("click", showUploadPlaceholder);
audioFile.addEventListener("change", () => {
  const file = audioFile.files[0];
  uploadName.textContent = file ? `${file.name} (${formatBytes(file.size)})` : "未选择文件";
});

loadSamples();

async function loadSamples() {
  try {
    const response = await fetch("/api/demo/samples");
    if (!response.ok) throw new Error("sample api unavailable");
    const payload = await response.json();
    renderSamples(payload.samples);
  } catch (error) {
    renderSamples(offlineSamples);
  }
}

function renderSamples(samples) {
  sampleList.innerHTML = "";
  samples.forEach((sample, index) => {
    const card = document.createElement("article");
    card.className = "sample-card";
    card.dataset.sampleId = sample.sample_id;
    card.innerHTML = `
      <div class="sample-header">
        <h3>${escapeHtml(sample.title)}</h3>
        <span class="label-chip label-${sample.expected_label}">${sample.expected_label}</span>
      </div>
      <p>${escapeHtml(sample.description)}</p>
      <audio controls preload="none" src="${escapeAttribute(sample.audio_url)}"></audio>
      <button class="primary-button" type="button">开始检测</button>
    `;
    card.querySelector("button").addEventListener("click", () => detectSample(sample.sample_id));
    sampleList.appendChild(card);
    if (index === 0) selectedSampleId = sample.sample_id;
  });
}

async function detectSample(sampleId) {
  selectedSampleId = sampleId;
  setSelectedSample(sampleId);
  renderLoading("正在读取预置样例结果...");
  await wait(260);
  try {
    const response = await fetch("/api/demo/predict-sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: sampleId })
    });
    if (!response.ok) throw new Error("prediction api unavailable");
    renderResult(await response.json());
  } catch (error) {
    renderResult(offlinePredictions[sampleId]);
  }
}

async function detectText() {
  const text = textInput.value.trim();
  if (!text) {
    renderError("请输入通话文本或 ASR 转写内容。");
    return;
  }
  renderLoading("正在执行文本分支检测...");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "text api unavailable");
    renderResult({ ...payload, mode: "text_demo" });
  } catch (error) {
    renderResult(buildOfflineTextPrediction(text));
  }
}

function showUploadPlaceholder() {
  const file = audioFile.files[0];
  if (!file) {
    renderError("请选择 mp3 或 wav 音频文件。");
    return;
  }
  renderResult({
    sample_id: "upload_placeholder",
    prediction: "normal",
    fraud_probability: 0,
    risk_level: "medium",
    asr_text: `已选择文件：${file.name}`,
    evidence: ["上传入口已保留", "答辩现场推荐使用预置样例保证稳定", "实时音频处理可接入 /api/predict"],
    model: "Whisper-small ASR + Chinese RoBERTa + MLP Fusion Classifier",
    mode: "upload_placeholder"
  });
}

function renderResult(result) {
  const isFraud = result.prediction === "fraud";
  const title = isFraud ? "疑似电诈" : "正常通话";
  const probability = Number(result.fraud_probability || 0);
  const percent = Math.max(3, Math.min(100, Math.round(probability * 100)));
  const modeLabel = result.mode === "text_demo" ? "文本分支演示" : "音频+文本融合演示";

  resultPanel.innerHTML = `
    <div class="status-line">${modeLabel}</div>
    <div class="risk-banner risk-${result.risk_level}">
      <strong>${title}</strong>
      <div class="probability">fraud probability: ${probability.toFixed(3)}</div>
    </div>
    <div class="risk-meter" aria-label="风险分数">
      <span style="width: ${percent}%"></span>
    </div>
    <div class="evidence-grid">
      <section class="evidence-box">
        <h3>ASR / 输入文本</h3>
        <p class="transcript">${escapeHtml(result.asr_text || "")}</p>
      </section>
      <section class="evidence-box">
        <h3>风险依据</h3>
        <ul>
          ${(result.evidence || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
        </ul>
      </section>
    </div>
    <div class="evidence-box" style="margin-top: 14px">
      <h3>模型路径</h3>
      <p class="transcript">${escapeHtml(result.model)}：音频与文本特征融合后输出 normal / fraud 二分类结果。</p>
    </div>
    <div class="result-actions">
      <button class="secondary-button" type="button" onclick="resetResult()">查看模型流程</button>
      <button class="secondary-button" type="button" onclick="copySummary()">复制结果摘要</button>
    </div>
  `;
}

function renderLoading(message) {
  resultPanel.innerHTML = `
    <div class="result-empty">
      <p class="eyebrow">Detecting</p>
      <h2>${escapeHtml(message)}</h2>
      <div class="flow-row">
        <span>音频输入</span>
        <span>Whisper-small</span>
        <span>RoBERTa</span>
        <span>MLP 融合分类</span>
        <span>输出结果</span>
      </div>
    </div>
  `;
}

function renderError(message) {
  resultPanel.innerHTML = `
    <div class="result-empty">
      <p class="eyebrow">Input Required</p>
      <h2>${escapeHtml(message)}</h2>
    </div>
  `;
}

function resetResult() {
  resultPanel.innerHTML = `
    <div class="result-empty">
      <p class="eyebrow">Model Flow</p>
      <h2>音频与文本融合检测</h2>
      <div class="flow-row">
        <span>音频输入</span>
        <span>Whisper-small</span>
        <span>RoBERTa</span>
        <span>MLP 融合分类</span>
        <span>normal / fraud</span>
      </div>
    </div>
  `;
}

async function copySummary() {
  const text = resultPanel.innerText.replace(/\n{2,}/g, "\n");
  if (navigator.clipboard) {
    await navigator.clipboard.writeText(text);
  }
}

function activateTab(name) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.remove("active"));
  document.querySelector(`#${name}-pane`).classList.add("active");
}

function setSelectedSample(sampleId) {
  document.querySelectorAll(".sample-card").forEach((card) => {
    card.classList.toggle("selected", card.dataset.sampleId === sampleId);
  });
}

function buildOfflineTextPrediction(text) {
  const keywords = ["安全账户", "转账", "验证码", "银行卡", "冻结", "涉嫌", "公检法", "贷款", "中奖"];
  const evidence = keywords.filter((keyword) => text.includes(keyword));
  const fraud = evidence.length > 0;
  return {
    sample_id: "text_input",
    prediction: fraud ? "fraud" : "normal",
    fraud_probability: fraud ? Math.min(0.72 + evidence.length * 0.08, 0.98) : 0.18,
    risk_level: fraud ? "high" : "low",
    asr_text: text,
    evidence: fraud ? evidence : ["未发现典型转账/验证码/冒充官方话术"],
    model: "Whisper-small ASR + Chinese RoBERTa + MLP Fusion Classifier",
    mode: "text_demo"
  };
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value);
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

window.resetResult = resetResult;
window.copySummary = copySummary;
