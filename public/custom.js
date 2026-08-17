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
