/* Groq key control, shared by the dashboard and the Attack Lab.
   - A small "Groq key" chip (bottom-left) shows the key health and opens the update dialog.
   - When Groq is rate-limited or rejects the key, a bar appears asking for a new key.
   - Updating the key hot-swaps it on the server; paused analysis continues by itself. */
(function () {
  if (window.__groqKeyUI) return; window.__groqKeyUI = true;

  const css = `
  #gk-chip{position:fixed;left:16px;bottom:16px;z-index:9998;display:flex;align-items:center;gap:8px;padding:8px 13px;border-radius:999px;
    background:#141820;color:#cbd5e1;font:500 11.5px/1 "JetBrains Mono",ui-monospace,Menlo,monospace;border:1px solid #1e2530;
    cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.5)}
  #gk-chip:hover{border-color:rgba(226,0,26,.5);color:#fff}
  #gk-chip i{width:8px;height:8px;border-radius:50%;background:#64748b}
  #gk-chip.ok i{background:#10b981;box-shadow:0 0 8px rgba(16,185,129,.6)} #gk-chip.rate_limited i{background:#f59e0b;box-shadow:0 0 8px rgba(245,158,11,.6)}
  #gk-chip.invalid i,#gk-chip.missing i{background:#ff3344;box-shadow:0 0 8px rgba(255,51,68,.6)}
  #gk-bar{position:fixed;left:50%;bottom:20px;transform:translateX(-50%);z-index:9999;display:none;max-width:min(720px,calc(100vw - 32px));
    align-items:center;gap:14px;padding:12px 16px;border-radius:12px;background:#1a1400;color:#fbbf24;border:1px solid rgba(245,158,11,.4);
    font:13px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;box-shadow:0 10px 36px rgba(0,0,0,.6)}
  #gk-bar.show{display:flex} #gk-bar.invalid,#gk-bar.missing{background:#1a0000;color:#fca5a5;border-color:rgba(255,51,68,.45)}
  #gk-bar button{flex:none;border:0;border-radius:8px;padding:8px 14px;font:600 12px Inter,system-ui,sans-serif;cursor:pointer;color:#fff;
    background:linear-gradient(135deg,#e2001a,#a5001a);box-shadow:0 0 16px rgba(226,0,26,.3)}
  #gk-modal{position:fixed;inset:0;z-index:10000;display:none;align-items:center;justify-content:center;background:rgba(5,6,8,.82);backdrop-filter:blur(8px);padding:20px;
    font:14px/1.55 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
  #gk-modal.show{display:flex}
  #gk-modal .box{width:min(460px,100%);background:#141820;color:#e2e8f0;border:1px solid #1e2530;border-top:3px solid #e2001a;border-radius:14px;padding:24px 26px;box-shadow:0 0 60px rgba(226,0,26,.15)}
  #gk-modal h2{margin:0 0 6px;font-size:18px;color:#fff} #gk-modal p{margin:0 0 14px;color:#94a3b8;font-size:13px}
  #gk-modal input{width:100%;padding:11px 12px;border:1px solid #1e2530;border-radius:9px;background:#0a0c0f;color:#e2e8f0;font:13px "JetBrains Mono",ui-monospace,Menlo,monospace}
  #gk-modal input:focus{outline:none;border-color:#e2001a;box-shadow:0 0 0 3px rgba(226,0,26,.18)}
  #gk-modal .row{display:flex;gap:10px;justify-content:flex-end;margin-top:16px}
  #gk-modal .row button{padding:9px 18px;border-radius:9px;font:600 12.5px Inter,system-ui,sans-serif;cursor:pointer;border:1px solid #1e2530;background:none;color:#94a3b8}
  #gk-modal .row button:hover{color:#fff;border-color:#64748b}
  #gk-modal .row .go{background:linear-gradient(135deg,#e2001a,#a5001a);color:#fff;border:0;box-shadow:0 0 16px rgba(226,0,26,.3)} #gk-modal .row .go:disabled{opacity:.5;cursor:wait}
  #gk-msg{min-height:20px;margin-top:10px;font-size:13px} #gk-msg.err{color:#ff6677} #gk-msg.ok{color:#34d399}
  #gk-modal small{display:block;margin-top:12px;color:#64748b;font-size:11.5px}`;
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);

  const root = document.createElement("div");
  root.innerHTML = `
    <button id="gk-chip" type="button" aria-label="Groq API key"><i></i><span>Groq key</span></button>
    <div id="gk-bar" role="alert"><span id="gk-bar-text"></span><button type="button" id="gk-bar-btn">Update key</button></div>
    <div id="gk-modal" role="dialog" aria-modal="true" aria-labelledby="gk-t"><div class="box">
      <h2 id="gk-t">Update the Groq API key</h2>
      <p id="gk-lead">Paste a new key from console.groq.com/keys. Analysis that was waiting on the old key carries on from where it stopped.</p>
      <input id="gk-input" type="password" autocomplete="off" spellcheck="false" placeholder="gsk_…" aria-label="Groq API key">
      <div id="gk-msg" aria-live="polite"></div>
      <div class="row"><button type="button" id="gk-cancel">Close</button><button type="button" class="go" id="gk-save">Update key</button></div>
      <small>The key is kept until the system restarts. After a restart the key in backend/config/.env is used again.</small>
    </div></div>`;
  document.body.appendChild(root);
  const $ = id => document.getElementById(id);

  function openModal(lead) {
    if (lead) $("gk-lead").textContent = lead;
    $("gk-msg").textContent = ""; $("gk-msg").className = ""; $("gk-input").value = "";
    $("gk-modal").classList.add("show"); setTimeout(() => $("gk-input").focus(), 50);
  }
  function closeModal() { $("gk-modal").classList.remove("show"); }

  async function save() {
    const key = $("gk-input").value.trim(); if (!key) return;
    $("gk-save").disabled = true; $("gk-msg").className = ""; $("gk-msg").textContent = "Checking the key with Groq…";
    try {
      const r = await fetch("/api/settings/groq", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "The key was not accepted.");
      $("gk-msg").className = "ok"; $("gk-msg").textContent = j.message;
      $("gk-input").value = ""; poll();
      setTimeout(closeModal, 1400);
    } catch (e) { $("gk-msg").className = "err"; $("gk-msg").textContent = e.message; }
    $("gk-save").disabled = false;
  }

  let last = "";
  async function poll() {
    try {
      const s = await (await fetch("/api/settings/groq", { cache: "no-store" })).json();
      const chip = $("gk-chip"); chip.className = s.state;
      chip.querySelector("span").textContent = "Groq key " + (s.key_hint || "not set") + (s.state === "ok" ? "" : s.state === "rate_limited" ? " · limit reached" : " · needs attention");
      const bar = $("gk-bar");
      if (s.state === "ok") { bar.classList.remove("show"); }
      else {
        bar.className = "show " + s.state;
        $("gk-bar-text").textContent = s.state === "rate_limited"
          ? "Groq's usage limit was reached" + (s.message ? " (" + s.message + ")" : "") + ". The AI experts wait and continue by themselves when it resets. Limits belong to your Groq account, so a new key only helps if it comes from a different account."
          : s.state === "invalid" ? "Groq rejected the API key, so the AI experts are paused. Paste a valid key and they will continue where they stopped."
          : "No Groq API key is set. Paste one so the AI experts can work.";
      }
      last = s.state;
    } catch (_) {}
  }

  $("gk-chip").onclick = () => openModal();
  $("gk-bar-btn").onclick = () => openModal();
  $("gk-cancel").onclick = closeModal; $("gk-save").onclick = save;
  $("gk-input").addEventListener("keydown", e => { if (e.key === "Enter") save(); if (e.key === "Escape") closeModal(); });
  poll(); setInterval(poll, 4000);
})();
