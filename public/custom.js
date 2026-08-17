// Adds a "Create an account" link under chainlit's login form.
(function () {
  function inject() {
    if (document.getElementById("gcf-signup-link")) return;
    var form = document.querySelector("form");
    if (!form || !/password/i.test(form.innerHTML)) return;
    var a = document.createElement("a");
    a.id = "gcf-signup-link";
    a.href = "/register";
    a.textContent = "No account? Create one";
    a.style.cssText = "display:block;text-align:center;margin-top:12px;font-size:14px;color:inherit;opacity:.8";
    form.appendChild(a);
  }
  new MutationObserver(inject).observe(document.documentElement, {childList: true, subtree: true});
  inject();
})();
