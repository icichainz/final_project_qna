(function () {
  "use strict";

  var form = document.getElementById("registration-form");
  if (!form) return;

  var username = document.getElementById("username");
  var password = document.getElementById("password");
  var confirmation = document.getElementById("confirm-password");
  var submitButton = document.getElementById("submit-button");
  var alertBox = document.getElementById("form-alert");
  var usernamePattern = /^[A-Za-z0-9][A-Za-z0-9_.@-]{2,63}$/;

  function showAlert(message, variant) {
    alertBox.textContent = message;
    alertBox.classList.toggle("success", variant === "success");
    alertBox.setAttribute("role", variant === "success" ? "status" : "alert");
    alertBox.setAttribute("aria-live", variant === "success" ? "polite" : "assertive");
    alertBox.hidden = !message;
  }

  function fieldError(input, message) {
    var error = document.getElementById(input.id + "-error");
    input.setAttribute("aria-invalid", message ? "true" : "false");
    if (error) error.textContent = message || "";
    return Boolean(message);
  }

  function validate() {
    var firstInvalid = null;
    var normalizedUsername = username.value.trim();
    var usernameMessage = "";
    var passwordMessage = "";
    var confirmationMessage = "";

    if (!normalizedUsername) {
      usernameMessage = "Enter a username.";
    } else if (!usernamePattern.test(normalizedUsername)) {
      usernameMessage = "Use 3–64 allowed characters and start with a letter or number.";
    }

    if (!password.value) {
      passwordMessage = "Enter a password.";
    } else if (password.value.length < 8) {
      passwordMessage = "Password must contain at least 8 characters.";
    }

    if (!confirmation.value) {
      confirmationMessage = "Confirm your password.";
    } else if (confirmation.value !== password.value) {
      confirmationMessage = "Passwords do not match.";
    }

    if (fieldError(username, usernameMessage)) firstInvalid = firstInvalid || username;
    if (fieldError(password, passwordMessage)) firstInvalid = firstInvalid || password;
    if (fieldError(confirmation, confirmationMessage)) firstInvalid = firstInvalid || confirmation;
    if (firstInvalid) firstInvalid.focus();
    return !firstInvalid;
  }

  function setBusy(busy) {
    Array.prototype.forEach.call(form.elements, function (element) {
      element.disabled = busy;
    });
    submitButton.setAttribute("aria-busy", busy ? "true" : "false");
    submitButton.querySelector(".button-label").textContent = busy ? "Creating account…" : "Create account";
  }

  Array.prototype.forEach.call(document.querySelectorAll("[data-password-toggle]"), function (button) {
    button.addEventListener("click", function () {
      var input = document.getElementById(button.getAttribute("data-password-toggle"));
      var reveal = input.type === "password";
      input.type = reveal ? "text" : "password";
      button.setAttribute("aria-pressed", reveal ? "true" : "false");
      button.setAttribute("aria-label", reveal ? "Hide password" : "Show password");
      input.focus();
    });
  });

  [username, password, confirmation].forEach(function (input) {
    input.addEventListener("input", function () {
      fieldError(input, "");
      showAlert("");
    });
  });

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    showAlert("");
    if (!validate()) return;

    var normalizedUsername = username.value.trim();
    setBusy(true);
    try {
      var response = await fetch("/register", {
        method: "POST",
        credentials: "same-origin",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username: normalizedUsername, password: password.value})
      });
      var data = await response.json().catch(function () { return {}; });
      if (!response.ok) throw new Error(data.detail || "Registration failed. Please try again.");

      showAlert("Account created. Signing you in…", "success");
      var loginBody = new FormData();
      loginBody.append("username", normalizedUsername);
      loginBody.append("password", password.value);
      var loginResponse = await fetch("/login", {
        method: "POST",
        credentials: "same-origin",
        body: loginBody
      }).catch(function () { return null; });

      if (loginResponse && loginResponse.ok) {
        window.location.assign("/");
        return;
      }
      showAlert("Account created. Redirecting you to sign in…", "success");
      window.setTimeout(function () { window.location.assign("/"); }, 1200);
    } catch (error) {
      showAlert(error instanceof Error ? error.message : "Registration failed. Please try again.");
      setBusy(false);
    }
  });
})();
