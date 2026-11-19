const $ = (id) => document.getElementById(id);
const src = $("src"), out = $("out"), source = $("source"), target = $("target"), model = $("model");
let quality = "fast", timer = null, controller = null, rtlByCode = {}, notes = {}, labels = {}, pollTimer = null, lastSource = null;

// ---------- models
async function loadModels() {
  const d = await (await fetch("/api/models")).json();
  model.innerHTML = d.models.map(m => `<option value="${m.key}">${m.label}</option>`).join("");
  d.models.forEach(m => { notes[m.key] = m.note; labels[m.key] = m.label; });
  const saved = localStorage.getItem("model");
  model.value = d.models.some(m => m.key === saved) ? saved : d.default;
  $("model-note").textContent = notes[model.value];
  await chooseModel(model.value);
}
async function chooseModel(key) {
  localStorage.setItem("model", key);
  $("model-note").textContent = notes[key];
  const s = await (await fetch("/api/model", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: key }) })).json();
  if (!s.ready && !s.error) {                      // only show the banner when we really have to wait
    source.disabled = target.disabled = true;
    banner(`Loading ${labels[key]}… The first time, it downloads from Hugging Face, which can take a few minutes. Next time it is much quicker.`);
  }
  handleStatus(s);
}
function markLoaded(loaded) {                      // ● = already in memory, switching is instant
  [...model.options].forEach(o => o.textContent = (loaded.includes(o.value) ? "● " : "") + labels[o.value]);
}
model.onchange = () => chooseModel(model.value);

// ---------- wait until the chosen model is ready
async function poll() {
  try { handleStatus(await (await fetch("/api/status")).json()); }
  catch (e) { clearTimeout(pollTimer); pollTimer = setTimeout(poll, 2000); }
}
function handleStatus(s) {
  clearTimeout(pollTimer);
  if (s.loaded) markLoaded(s.loaded);
  if (s.model !== model.value) return;
  if (s.error) { banner(`This Transformer could not load: ${s.error}`, true); return; }
  if (!s.ready) { pollTimer = setTimeout(poll, 1500); return; }
  fillLanguages(s.languages, s.detection);
  banner(null);
  if (src.value.trim()) translate();
}
function banner(msg, error = false) {
  const b = $("banner");
  if (!msg) { b.classList.add("hidden"); return; }
  b.textContent = msg; b.classList.remove("hidden"); b.classList.toggle("error", error);
}
function fillLanguages(langs, detection) {
  const has = c => langs.some(l => l.code === c);
  const keepS = source.value || localStorage.getItem("src"), keepT = target.value || localStorage.getItem("tgt");
  const opts = langs.map(l => `<option value="${l.code}">${l.name}</option>`).join("");
  source.innerHTML = (detection ? `<option value="auto">Detect language</option>` : "") + opts;
  target.innerHTML = opts;
  rtlByCode = {}; langs.forEach(l => rtlByCode[l.code] = l.rtl);
  source.value = (keepS === "auto" && detection) || has(keepS) ? keepS : (detection ? "auto" : "eng_Latn");
  target.value = has(keepT) ? keepT : "arb_Arab";
  source.disabled = target.disabled = false;
  setSourceDir();
}
function setSourceDir() { src.classList.toggle("rtl", !!rtlByCode[source.value]); }

// ---------- translation
async function translate() {
  const text = src.value;
  $("count").textContent = `${text.length} / 5000`;
  if (!text.trim()) {
    out.innerHTML = '<span class="placeholder">Translation</span>';
    $("status").textContent = ""; $("detected").textContent = ""; return;
  }
  if (source.disabled) return;
  if (controller) controller.abort();               // drop the outdated request
  controller = new AbortController();
  out.classList.add("busy"); $("status").textContent = "Translating…";
  try {
    const r = await fetch("/api/translate", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
      body: JSON.stringify({ text, source: source.value, target: target.value, quality, model: model.value }),
    });
    const d = await r.json();
    if (d.error) { $("status").textContent = d.error; return; }
    out.textContent = d.translation;
    scheduleScore();
    out.classList.toggle("rtl", d.rtl);
    const name = d.detected && source.querySelector(`option[value="${d.detected}"]`)?.textContent;
    $("detected").textContent = d.detected ? `Detected: ${name || d.detected}` : "";
    src.classList.toggle("rtl", !!rtlByCode[d.source]);
    lastSource = d.source;
    $("status").textContent = `${d.seconds}s · ${d.sentences} sentence${d.sentences === 1 ? "" : "s"}`;
  } catch (e) {
    if (e.name !== "AbortError") $("status").textContent = "Could not reach the server.";
  } finally { out.classList.remove("busy"); }
}
function schedule() {
  $("count").textContent = `${src.value.length} / 5000`;
  if (!$("live").checked) return;
  clearTimeout(timer);
  timer = setTimeout(translate, 350);               // wait for a short pause in typing
}

// ---------- controls
src.addEventListener("input", schedule);
src.addEventListener("keydown", e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) translate(); });
$("go").onclick = translate;
$("clear").onclick = () => { src.value = ""; translate(); src.focus(); };
$("copy").onclick = async () => {
  await navigator.clipboard.writeText(out.textContent);
  $("copy").textContent = "Copied"; setTimeout(() => $("copy").textContent = "Copy", 1200);
};
$("swap").onclick = () => {
  if (source.disabled) return;
  // With "Detect language", swap using the language that was actually detected
  const from = source.value === "auto" ? (lastSource || "eng_Latn") : source.value;
  const to = target.value;
  if (src.value.trim() && !out.querySelector(".placeholder") && !out.classList.contains("busy")) {
    src.value = out.textContent;
  }
  source.value = to; target.value = from;
  const btn = $("swap"); btn.classList.remove("spin"); void btn.offsetWidth; btn.classList.add("spin");
  onLangChange();
};
function onLangChange() {
  localStorage.setItem("src", source.value); localStorage.setItem("tgt", target.value);
  setSourceDir(); translate();
}
source.onchange = target.onchange = onLangChange;
document.querySelectorAll(".seg button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".seg button").forEach(x => x.classList.remove("active"));
  b.classList.add("active"); quality = b.dataset.q; translate();
});

// ---------- BLEU and chrF (computed by metrics.py on the server)
const ref = $("ref");
let scoreTimer = null;
function scheduleScore() { clearTimeout(scoreTimer); scoreTimer = setTimeout(scoreNow, 300); }
function resetScores(msg) {
  $("bleu").textContent = "–"; $("chrf").textContent = "–";
  $("bleu-detail").textContent = "word n-gram overlap (1–4)";
  $("chrf-detail").textContent = "character n-gram F-score (1–6)";
  $("score-msg").textContent = msg || "";
}
async function scoreNow() {
  const hyp = out.querySelector(".placeholder") ? "" : out.textContent;
  if (!ref.value.trim() || !hyp.trim()) { resetScores(ref.value.trim() ? "Translate some text first." : ""); return; }
  try {
    const r = await fetch("/api/score", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hypothesis: hyp, reference: ref.value }) });
    const d = await r.json();
    if (d.error) { resetScores(d.error); return; }
    $("bleu").textContent = d.bleu.toFixed(1);
    $("chrf").textContent = d.chrf.toFixed(1);
    $("bleu-detail").textContent = `precisions ${d.bleu_precisions.join(" / ")} · BP ${d.brevity_penalty}`;
    $("chrf-detail").textContent = `precision ${d.chrf_precision} · recall ${d.chrf_recall}`;
    let note = d.settings;
    if (d.bleu === 0 && d.chrf > 0) {
      note = "BLEU is 0 because no whole word of the translation matches the reference exactly " +
             "(BLEU counts exact words; chrF gives partial credit for similar words). " +
             "Check the reference is in the target language and spelled the same way.";
    } else if (d.bleu === 0) {
      note = "No match at all: check that the reference is a translation in the target language.";
    }
    $("score-msg").textContent = note;
  } catch (e) { resetScores("Could not reach the server."); }
}
ref.addEventListener("input", scheduleScore);

loadModels();
