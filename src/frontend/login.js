
const GOOGLE_CLIENT_ID = "516200553822-h55harjhv2nlui8stfqon6tjnh521501.apps.googleusercontent.com";

const googleLoginBtn = document.getElementById("google-login-btn");
const loginMessage = document.getElementById("login-message");

googleLoginBtn.addEventListener("click", () => {
    if (!window.google?.accounts?.oauth2) {
        loginMessage.textContent =
            "Google sign-in library could not load. Check your internet connection and reload.";
        return;
    }

    loginMessage.textContent = "Connecting to Google...";

    const client = window.google.accounts.oauth2.initCodeClient({
        client_id: GOOGLE_CLIENT_ID,
        scope: "openid email profile",
        ux_mode: "popup",
        callback: (response) => {
            if (response.error || !response.code) {
                loginMessage.textContent =
                    "Google authorization failed. Please try again.";
                console.error("Google OAuth error:", response);
                return;
            }

            loginMessage.textContent =
             "Google authorization demo completed. Login is not enabled yet.";

            console.log("Google authorization code received.");
            // Next step: send response.code to your FastAPI backend
            // for secure exchange and identity verification.
        },
        error_callback: (error) => {
            loginMessage.textContent =
                "Google sign-in popup failed. Please allow popups and try again.";
            console.error("Google popup error:", error);
        }
    });

    client.requestCode();
});
