const searchForm = document.querySelector("#search-form");
const questionInput = document.querySelector("#question");
const knownIdInput = document.querySelector("#known-id");
const questionOptions = document.querySelector("#saved-questions");
const message = document.querySelector("#form-message");
const workspace = document.querySelector("#workspace");
const emptyState = document.querySelector("#empty-state");
const resultList = document.querySelector("#result-list");
const resultCount = document.querySelector("#result-count");
const resultsQuery = document.querySelector("#results-query");
const modeDescription = document.querySelector("#mode-description");
const resultHint = document.querySelector("#result-hint");
const comparisonList = document.querySelector("#comparison-list");
const expectedDocument = document.querySelector("#expected-document");
const modeButtons = [...document.querySelectorAll("[data-mode]")];

const modeDetails = {
  keyword: {label: "Keyword", description: "Matches the words in the question", symbol: "Aa"},
  vector: {label: "Vector", description: "Looks for similar meaning", symbol: "◌"},
  hybrid: {label: "Hybrid", description: "Combines text and vector search", symbol: "✳"},
  exact: {label: "Exact ID", description: "Looks up the ID you provide", symbol: "#"},
};

const state = {data: null, query: null, knownId: "", mode: "hybrid"};

function element(tagName, className, text) {
  const node = document.createElement(tagName);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function documentIcon() {
  const wrapper = element("span", "document-icon");
  wrapper.setAttribute("aria-hidden", "true");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  for (const shape of ["M6 3.8h8.5L19 8.3v11.9H6z", "M14.5 3.8v4.5H19", "M9 12h7M9 15h7"]) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", shape);
    svg.append(path);
  }
  wrapper.append(svg);
  return wrapper;
}

function getExactCheck() {
  if (!state.knownId) return {kind: "empty", returnedIds: []};
  if (!Object.hasOwn(state.data.exact_checks, state.knownId)) {
    return {kind: "unrecorded", returnedIds: []};
  }
  const returnedIds = state.data.exact_checks[state.knownId];
  return {kind: returnedIds.length ? "found" : "missing", returnedIds};
}

function exactMessage(check) {
  if (check.kind === "empty") return "Enter a known document ID above to replay an exact lookup.";
  if (check.kind === "unrecorded") return "This ID was not included in the recorded exact-lookup examples.";
  if (check.kind === "missing") return "The recorded exact lookup returned no document for this case-sensitive ID.";
  return "";
}

function displayText(documentId) {
  const text = state.data.documents[documentId];
  const prefix = `${documentId}: `;
  return text.startsWith(prefix) ? text.slice(prefix.length) : text;
}

function renderResults() {
  const exact = getExactCheck();
  const ids = state.mode === "exact" ? exact.returnedIds : state.query.modes[state.mode].returned_ids;
  modeDescription.textContent = modeDetails[state.mode].description;
  resultCount.textContent = state.mode === "exact" && ["empty", "unrecorded"].includes(exact.kind)
    ? "No saved lookup selected"
    : `${ids.length} recorded ${ids.length === 1 ? "result" : "results"}`;

  resultList.replaceChildren();
  if (!ids.length) {
    const reason = state.mode === "exact" ? exactMessage(exact) : "No results were returned in the saved example.";
    resultList.append(element("div", "result-empty", reason));
  }

  ids.forEach((documentId, index) => {
    const expected = state.query.relevant.includes(documentId);
    const card = element("article", expected ? "result-card is-expected" : "result-card");
    const content = element("div", "document-content");
    const top = element("div", "document-top");
    const title = element("div", "document-title");
    title.append(element("span", "", state.mode === "exact" ? "ID" : `#${index + 1}`));
    title.append(element("strong", "", documentId));
    top.append(title, element("span", expected ? "document-badge is-expected" : "document-badge",
      expected ? "EXPECTED NOTE" : "OTHER ID"));
    content.append(top, element("p", "document-copy", displayText(documentId)));
    card.append(documentIcon(), content);
    resultList.append(card);
  });

  if (state.mode === "exact") {
    resultHint.textContent = "Exact lookup uses the case-sensitive ID you provide, not the words in the question.";
  } else if (state.query.kind === "exact_identifier") {
    resultHint.textContent = "Similar-looking IDs can refer to different notes. Exact lookup needs the known ID.";
  } else {
    resultHint.textContent = "Ranked results are a starting point. Review the note before sharing an answer.";
  }
}

function comparisonStatus(mode) {
  if (mode !== "exact") {
    const rank = state.query.modes[mode].rank;
    return rank ? {text: `Rank #${rank}`, tone: rank === 1 ? "is-good" : ""}
      : {text: `Not in top ${state.data.top_k}`, tone: "is-miss"};
  }
  const check = getExactCheck();
  if (check.kind === "empty") return {text: "Enter an ID", tone: ""};
  if (check.kind === "unrecorded") return {text: "Not in replay", tone: ""};
  if (check.kind === "missing") return {text: "No document", tone: "is-miss"};
  return check.returnedIds.some((documentId) => state.query.relevant.includes(documentId))
    ? {text: "Found exactly", tone: "is-good"}
    : {text: "Different ID found", tone: "is-miss"};
}

function renderComparison() {
  expectedDocument.textContent = state.query.relevant.join(", ");
  comparisonList.replaceChildren();
  for (const mode of ["keyword", "vector", "hybrid", "exact"]) {
    const details = modeDetails[mode];
    const status = comparisonStatus(mode);
    const row = element("div", "comparison-row");
    const name = element("div", "comparison-name");
    const icon = element("span", `comparison-icon ${mode}`, details.symbol);
    icon.setAttribute("aria-hidden", "true");
    const label = element("span", "comparison-label");
    label.append(element("strong", "", details.label), element("span", "", details.description));
    name.append(icon, label);
    row.append(name, element("span", `comparison-status ${status.tone}`, status.text));
    comparisonList.append(row);
  }
}

function render() {
  resultsQuery.textContent = `Recorded results for “${state.query.question}”.`;
  for (const button of modeButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode));
  }
  renderResults();
  renderComparison();
}

function showQuestion(question) {
  state.query = question;
  state.knownId = knownIdInput.value.trim();
  message.textContent = "";
  workspace.hidden = false;
  emptyState.hidden = true;
  render();
}

function chooseExample(queryId) {
  const question = state.data.queries.find((candidate) => candidate.id === queryId);
  if (!question) return;
  questionInput.value = question.question;
  knownIdInput.value = question.known_id;
  showQuestion(question);
}

searchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!state.data) return;
  const question = state.data.queries.find((candidate) => candidate.question === questionInput.value.trim());
  if (!question) {
    state.query = null;
    workspace.hidden = true;
    emptyState.hidden = false;
    emptyState.textContent = "This offline preview only replays recorded questions. Choose an example or a question from the list.";
    message.textContent = "No saved results for that question. No AWS request was sent.";
    return;
  }
  showQuestion(question);
});

knownIdInput.addEventListener("input", () => {
  if (!state.query) return;
  state.knownId = knownIdInput.value.trim();
  render();
});

for (const button of modeButtons) {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    if (state.query) render();
  });
}

for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", () => {
    if (state.data) chooseExample(button.dataset.example);
  });
}

async function initialize() {
  try {
    const response = await fetch("/demo-data.json", {cache: "no-store"});
    if (!response.ok) throw new Error("Recorded examples could not be loaded.");
    state.data = await response.json();
    for (const question of state.data.queries) {
      const option = element("option");
      option.value = question.question;
      questionOptions.append(option);
    }
    document.querySelector("#document-count").textContent = `${Object.keys(state.data.documents).length} synthetic notes`;
    chooseExample("identifier-01");
  } catch (error) {
    emptyState.textContent = "The offline data could not be loaded. Start the demo with python demo/server.py.";
    message.textContent = "No AWS request was sent.";
    console.error(error);
  }
}

initialize();
