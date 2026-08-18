// Login-page helper: adds a "Create an account" link near the password field.
// Robust to chainlit's React DOM: anchors on input[type=password], not <form>.
(function () {
  function inject() {
    if (document.getElementById("gcf-signup-link")) return;
    var pw = document.querySelector('input[type="password"]');
    if (!pw) return;
    // only the login screen (register page has its own markup)
    if (location.pathname !== "/" && !location.pathname.startsWith("/login")) return;
    var host = pw.closest("form") || pw.parentElement && pw.parentElement.parentElement;
    if (!host) return;
    var a = document.createElement("a");
    a.id = "gcf-signup-link";
    a.href = "/register";
    a.textContent = "No account? Create one · Pas de compte ? Créez-en un";
    a.style.cssText = "display:block;text-align:center;margin-top:14px;font-size:13px;color:inherit;opacity:.75;text-decoration:underline";
    host.appendChild(a);
  }
  new MutationObserver(inject).observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener("DOMContentLoaded", inject);
  inject();
})();

// Auto-resume: a page RELOAD at "/" (mid-chat refresh) jumps back into the
// most recent thread instead of a blank new chat. Fresh navigations and the
// New Chat button are unaffected; reloads on /thread/<id> already re-render.
(async function autoResume() {
  try {
    var nav = performance.getEntriesByType("navigation")[0];
    if (!nav || nav.type !== "reload") return;
    if (location.pathname !== "/") return;
    var r = await fetch("/project/threads", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({pagination: {first: 1}, filter: {}})
    });
    if (!r.ok) return;
    var d = await r.json();
    var t = d && d.data && d.data[0];
    if (t && t.id) location.replace("/thread/" + t.id);
  } catch (e) { /* fall through to the normal new-chat page */ }
})();
