// Front-end logic for the app. Plain JS, no build step.
(() => {
  const $ = (sel) => document.querySelector(sel);
  const RING_CIRC = 2 * Math.PI * 52; // r=52 in the SVG

  const state = { model: "deep" };

  const el = {
    complaint: $("#complaint"),
    examples: $("#examples"),
    analyze: $("#analyzeBtn"),
    btnLabel: $(".btn-label"),
    spinner: $(".spinner"),
    modelPicker: $("#modelPicker"),
    empty: $("#emptyState"),
    body: $("#resultBody"),
    triage: $("#triageBanner"),
    ring: $("#ringFg"),
    confNum: $("#confNum"),
    predLabel: $("#predLabel"),
    modelUsed: $("#modelUsed"),
    bars: $("#topBars"),
    narrative: $("#narrative"),
    highlightBlock: $("#highlightBlock"),
    compare: $("#compareRow"),
    status: $("#statusText"),
  };

  const MODEL_NAMES = { deep: "Deep TextCNN", classical: "Classical TF-IDF and LogReg", naive: "Naive baseline" };

  // examples
  async function loadExamples() {
    try {
      const res = await fetch("/api/examples");
      const items = await res.json();
      el.examples.innerHTML = "";
      items.forEach((ex) => {
        const b = document.createElement("button");
        b.className = "chip";
        b.textContent = ex.title;
        b.addEventListener("click", () => {
          el.complaint.value = ex.text;
          el.complaint.focus();
        });
        el.examples.appendChild(b);
      });
    } catch (_) { /* non-fatal */ }
  }

  // model picker
  el.modelPicker.querySelectorAll(".seg").forEach((btn) => {
    btn.addEventListener("click", () => {
      el.modelPicker.querySelectorAll(".seg").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.model = btn.dataset.model;
      if (el.complaint.value.trim().length > 14) analyze();
    });
  });

  // analyze
  async function analyze() {
    const text = el.complaint.value.trim();
    if (text.length < 15) { flashError("Please enter a longer complaint (at least a sentence)."); return; }
    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: state.model }),
      });
      const data = await res.json();
      if (!res.ok) { flashError(data.error || "Something went wrong."); return; }
      render(data, text);
    } catch (e) {
      flashError("Could not reach the model service.");
    } finally {
      setLoading(false);
    }
  }

  function setLoading(on) {
    el.analyze.disabled = on;
    el.spinner.hidden = !on;
    el.btnLabel.textContent = on ? "Checking" : "Check it";
  }

  function flashError(msg) {
    el.empty.hidden = true;
    el.body.hidden = true;
    let box = $("#errBox");
    if (!box) {
      box = document.createElement("div");
      box.id = "errBox";
      box.className = "error-msg";
      el.body.parentNode.insertBefore(box, el.body);
    }
    box.textContent = msg;
    box.hidden = false;
    setTimeout(() => { box.hidden = true; }, 4000);
  }

  // rendering
  const pct = (x) => `${Math.round(x * 100)}%`;

  function render(data, text) {
    const errBox = $("#errBox"); if (errBox) errBox.hidden = true;
    el.empty.hidden = true;
    el.body.hidden = false;

    // triage banner
    const tier = data.triage?.tier || "moderate";
    el.triage.className = `triage-banner tier-${tier}`;
    el.triage.innerHTML = `<span class="tier-tag">${tier}</span><span>${data.triage?.routing || ""}</span>`;

    // confidence ring
    const conf = data.confidence || 0;
    el.ring.style.strokeDasharray = RING_CIRC;
    el.ring.style.strokeDashoffset = RING_CIRC * (1 - conf);
    el.ring.style.stroke = tier === "critical" ? "#a81f1a" : tier === "high" ? "#9c5410" : "#4d7a1e";
    el.confNum.textContent = pct(conf);
    el.predLabel.textContent = data.prediction;
    el.modelUsed.textContent = `via ${MODEL_NAMES[data.model] || data.model}`;

    // top-k bars
    el.bars.innerHTML = "";
    (data.top_k || []).forEach((row, i) => {
      const div = document.createElement("div");
      div.className = "bar-row" + (i === 0 ? "" : " dim");
      div.innerHTML = `<span class="bar-name" title="${row.label}">${row.label}</span>
        <span class="bar-track"><span class="bar-fill"></span></span>
        <span class="bar-val">${pct(row.prob)}</span>`;
      el.bars.appendChild(div);
      requestAnimationFrame(() => { div.querySelector(".bar-fill").style.width = pct(row.prob); });
    });

    // narrative with word-level highlights
    if (data.highlights && data.highlights.length) {
      el.highlightBlock.hidden = false;
      el.narrative.innerHTML = highlight(text, data.highlights);
    } else {
      el.highlightBlock.hidden = true;
    }

    // model comparison
    el.compare.innerHTML = "";
    const order = ["naive", "classical", "deep"];
    order.filter((k) => data.compare && data.compare[k]).forEach((k) => {
      const c = data.compare[k];
      const div = document.createElement("div");
      div.className = "cmp" + (k === data.model ? " deployed" : "");
      div.innerHTML = `<div class="cmp-model">${k}</div>
        <div class="cmp-pred">${c.prediction}</div>
        <div class="cmp-conf">${pct(c.confidence)}</div>
        ${k === data.model ? '<div class="cmp-badge">shown above</div>' : '<div class="cmp-badge">&nbsp;</div>'}`;
      el.compare.appendChild(div);
    });
  }

  // wrap the strongest tokens in the narrative with a colour by weight
  function highlight(text, highlights) {
    const maxW = Math.max(...highlights.map((h) => h.weight), 1e-6);
    const weightOf = {};
    highlights.forEach((h) => { weightOf[h.token.toLowerCase()] = h.weight; });
    const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    return text.split(/(\s+)/).map((word) => {
      const key = word.toLowerCase().replace(/[^a-z0-9/\-]/g, "");
      const w = weightOf[key];
      if (w && w > 0) {
        const alpha = 0.12 + 0.42 * (w / maxW);
        return `<span class="tok" style="background: rgba(191,74,28,${alpha.toFixed(2)}); color:#221d15; font-weight:600">${esc(word)}</span>`;
      }
      return esc(word);
    }).join("");
  }

  el.analyze.addEventListener("click", analyze);
  el.complaint.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
  });

  loadExamples();
})();
