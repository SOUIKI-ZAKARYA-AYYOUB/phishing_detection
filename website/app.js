const samples = {
  phishing: {
    subject: "Action required: verify your mailbox immediately",
    body: `Dear user,

Your account storage is full and outgoing mail will be suspended today. Visit https://login-security.account-update.example.verify-mail.com/reset now to confirm your password.

Failure to act immediately will close access to payroll documents. This is a final warning.`
  },
  legitimate: {
    subject: "Thursday project sync notes",
    body: `Hi team,

Thanks for the discussion today. I added the model evaluation notes to the shared project folder and updated the task list for next week.

No action is required before Monday. Please review the notebook comments when you have time.`
  }
};

const subjectInput = document.querySelector("#emailSubject");
const bodyInput = document.querySelector("#emailBody");
const form = document.querySelector("#emailForm");
const analyzeButton = document.querySelector("#analyzeButton");
const clearButton = document.querySelector("#clearButton");

const output = {
  modelStatus: document.querySelector("#modelStatus"),
  modelStatusText: document.querySelector("#modelStatusText"),
  settingsLine: document.querySelector("#settingsLine"),
  decisionBlock: document.querySelector("#decisionBlock"),
  decisionLabel: document.querySelector("#decisionLabel"),
  fusionScore: document.querySelector("#fusionScore"),
  decisionText: document.querySelector("#decisionText"),
  scoreAValue: document.querySelector("#scoreAValue"),
  scoreBValue: document.querySelector("#scoreBValue"),
  scoreFusionValue: document.querySelector("#scoreFusionValue"),
  scoreABar: document.querySelector("#scoreABar"),
  scoreBBar: document.querySelector("#scoreBBar"),
  scoreFusionBar: document.querySelector("#scoreFusionBar"),
  thresholdValue: document.querySelector("#thresholdValue"),
  tokenCount: document.querySelector("#tokenCount"),
  urlCount: document.querySelector("#urlCount"),
  emailCount: document.querySelector("#emailCount"),
  ipCount: document.querySelector("#ipCount"),
  htmlCount: document.querySelector("#htmlCount"),
  punctuationCount: document.querySelector("#punctuationCount"),
  uppercaseRatio: document.querySelector("#uppercaseRatio"),
  suspiciousCount: document.querySelector("#suspiciousCount"),
  subdomainCount: document.querySelector("#subdomainCount"),
  reasonList: document.querySelector("#reasonList")
};

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function setStatus(type, text) {
  output.modelStatus.dataset.status = type;
  output.modelStatusText.textContent = text;
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  analyzeButton.textContent = isLoading ? "Analyzing..." : "Analyze email";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "The model server returned an error.");
  }
  return data;
}

async function loadStatus() {
  try {
    const status = await fetchJson("/api/status");
    const fusion = status.fusion;
    setStatus("ready", `Models loaded from ${status.models_path}`);
    output.settingsLine.textContent =
      `Fusion uses ${formatPercent(fusion.w_a)} structural, ${formatPercent(fusion.w_b)} text, threshold ${formatPercent(fusion.threshold)}.`;
  } catch (error) {
    setStatus("error", "Model server is not available. Start the website with: python website/server.py");
    output.settingsLine.textContent = error.message;
  }
}

async function analyzeEmail() {
  const subject = subjectInput.value.trim();
  const body = bodyInput.value.trim();

  if (!subject && !body) {
    renderEmptyState("Paste an email subject or body to run the models.");
    return;
  }

  setLoading(true);
  try {
    const result = await fetchJson("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ subject, body })
    });
    renderResult(result);
    setStatus("ready", "Models are running.");
  } catch (error) {
    setStatus("error", "Analysis failed.");
    renderError(error.message);
  } finally {
    setLoading(false);
  }
}

function renderEmptyState(message) {
  output.decisionBlock.className = "decision-block";
  output.decisionLabel.textContent = "Ready";
  output.fusionScore.textContent = "--";
  output.decisionText.textContent = message;
  updateMeter(output.scoreABar, 0);
  updateMeter(output.scoreBBar, 0);
  updateMeter(output.scoreFusionBar, 0);
  output.scoreAValue.textContent = "--";
  output.scoreBValue.textContent = "--";
  output.scoreFusionValue.textContent = "--";
  output.thresholdValue.textContent = "--";
  output.tokenCount.textContent = "--";
  resetFeatures();
  renderReasons(["The result will appear here after the model server analyzes the message."]);
}

function renderError(message) {
  output.decisionBlock.className = "decision-block error";
  output.decisionLabel.textContent = "Error";
  output.fusionScore.textContent = "--";
  output.decisionText.textContent = message;
  renderReasons(["Check that the local server is running and that every file in the models folder is present."]);
}

function renderResult(result) {
  const isPhishing = result.decision === "phishing";
  const scores = result.scores;
  const features = result.features;

  output.decisionBlock.className = `decision-block ${isPhishing ? "risky" : "safe"}`;
  output.decisionLabel.textContent = isPhishing ? "Phishing risk" : "Likely legitimate";
  output.fusionScore.textContent = formatPercent(scores.fusion);
  output.decisionText.textContent = isPhishing
    ? "The combined score is above the saved threshold. Verify the sender before taking action."
    : "The combined score is below the saved threshold. Keep normal caution with links and attachments.";

  output.scoreAValue.textContent = formatPercent(scores.model_a);
  output.scoreBValue.textContent = formatPercent(scores.model_b);
  output.scoreFusionValue.textContent = formatPercent(scores.fusion);
  output.thresholdValue.textContent = formatPercent(scores.threshold);
  output.tokenCount.textContent = String(result.tokens_used);

  updateMeter(output.scoreABar, scores.model_a);
  updateMeter(output.scoreBBar, scores.model_b);
  updateMeter(output.scoreFusionBar, scores.fusion);

  output.urlCount.textContent = String(features.num_urls);
  output.emailCount.textContent = String(features.num_emails);
  output.ipCount.textContent = String(features.num_ips);
  output.htmlCount.textContent = String(features.num_html_tags);
  output.punctuationCount.textContent = String(features.num_exclamations + features.num_questions);
  output.uppercaseRatio.textContent = formatPercent(features.uppercase_ratio);
  output.suspiciousCount.textContent = String(features.suspicious_word_count);
  output.subdomainCount.textContent = String(features.num_subdomains);

  renderReasons(result.reasons);
}

function resetFeatures() {
  [
    output.urlCount,
    output.emailCount,
    output.ipCount,
    output.htmlCount,
    output.punctuationCount,
    output.uppercaseRatio,
    output.suspiciousCount,
    output.subdomainCount
  ].forEach((item) => {
    item.textContent = "--";
  });
}

function renderReasons(reasons) {
  output.reasonList.replaceChildren();
  reasons.forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reason;
    output.reasonList.append(item);
  });
}

function updateMeter(element, value) {
  const width = Math.max(0, Math.min(100, Number(value) * 100));
  element.style.width = `${width}%`;
}

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    const sample = samples[button.dataset.sample];
    subjectInput.value = sample.subject;
    bodyInput.value = sample.body;
    analyzeEmail();
  });
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  analyzeEmail();
});

clearButton.addEventListener("click", () => {
  subjectInput.value = "";
  bodyInput.value = "";
  renderEmptyState("Paste an email subject or body to run the models.");
});

subjectInput.value = samples.phishing.subject;
bodyInput.value = samples.phishing.body;
renderEmptyState("The phishing sample is ready. Click Analyze email to run the saved models.");
loadStatus();
