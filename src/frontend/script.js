
console.log("AI RESEARCH AGENT SCRIPT LOADED");

const API_URL = "http://127.0.0.1:8000";
const CHAT_URL = `${API_URL}/chat`;
const UPLOAD_URL = `${API_URL}/upload-pdf`;

// Main sections
const chatSection = document.getElementById("chat-section");
const pdfSection = document.getElementById("pdf-section");
const imagesSection = document.getElementById("images-section");
const settingsSection = document.getElementById("settings-section");

// Chat elements
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendButton = chatForm.querySelector('button[type="submit"]');
const welcomeArea = document.getElementById("welcome-area");

// PDF attachment elements
const addPdfBtn = document.getElementById("add-pdf-btn");
const chatPdfUpload = document.getElementById("chat-pdf-upload");
const selectedPdf = document.getElementById("selected-pdf");

// Navigation
const navItems = document.querySelectorAll(".nav-item");
const newChatBtn = document.getElementById("new-chat-btn");

let selectedFile = null;
let isSending = false;

// --------------------------------------------------
// Navigation
// --------------------------------------------------

function showSection(section, activeButton = null) {
    [chatSection, pdfSection, imagesSection, settingsSection].forEach(item => {
        item.hidden = item !== section;
    });

    navItems.forEach(button => {
        button.classList.toggle("active", button === activeButton);
    });
}

navItems.forEach(button => {
    button.addEventListener("click", () => {
        const targetSection = document.getElementById(button.dataset.section);

        if (targetSection) {
            showSection(targetSection, button);
        }
    });
});

// --------------------------------------------------
// Chat messages
// --------------------------------------------------

function addMessage(message, className) {
    const messageElement = document.createElement("div");
    messageElement.className = className;
    messageElement.textContent = message;

    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageElement;
}

function showWelcome() {
    welcomeArea.hidden = chatMessages.children.length > 0;
}

// --------------------------------------------------
// PDF attachment in composer
// --------------------------------------------------

function renderSelectedPdf() {
    selectedPdf.replaceChildren();

    if (!selectedFile) {
        return;
    }

    const attachment = document.createElement("div");
    attachment.className = "selected-pdf-card";

    const icon = document.createElement("span");
    icon.className = "selected-pdf-icon";
    icon.textContent = "▤";

    const details = document.createElement("div");
    details.className = "selected-pdf-details";

    const name = document.createElement("span");
    name.className = "selected-pdf-name";
    name.textContent = selectedFile.name;
    name.title = selectedFile.name;

    const size = document.createElement("span");
    size.className = "selected-pdf-size";
    size.textContent = `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB · PDF`;

    details.append(name, size);

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "remove-pdf-button";
    removeButton.textContent = "×";
    removeButton.title = "Remove attached PDF";
    removeButton.setAttribute("aria-label", "Remove attached PDF");

    removeButton.addEventListener("click", () => {
        selectedFile = null;
        chatPdfUpload.value = "";
        renderSelectedPdf();
    });

    attachment.append(icon, details, removeButton);
    selectedPdf.appendChild(attachment);
}

addPdfBtn.addEventListener("click", () => {
    if (!isSending) {
        chatPdfUpload.click();
    }
});

chatPdfUpload.addEventListener("change", () => {
    const file = chatPdfUpload.files[0];

    if (!file) {
        return;
    }

    if (
        file.type !== "application/pdf" &&
        !file.name.toLowerCase().endsWith(".pdf")
    ) {
        alert("Please select a PDF file.");
        chatPdfUpload.value = "";
        return;
    }

    if (file.size === 0) {
        alert("The selected PDF is empty.");
        chatPdfUpload.value = "";
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        alert("PDF must be 20 MB or smaller.");
        chatPdfUpload.value = "";
        return;
    }

    selectedFile = file;
    renderSelectedPdf();
});

// Upload the selected PDF only when Send is clicked.
async function uploadSelectedPdf(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(UPLOAD_URL, {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || `PDF upload failed: ${response.status}`);
    }

    console.log("PDF uploaded and indexed:", data);
    return data;
}

// --------------------------------------------------
// Chat submission: PDF first, then question
// --------------------------------------------------

chatForm.addEventListener("submit", async event => {
    event.preventDefault();

    if (isSending) {
        return;
    }

    const question = questionInput.value.trim();

    if (!question) {
        questionInput.focus();
        return;
    }

    const fileToUpload = selectedFile;

    isSending = true;
    questionInput.disabled = true;
    sendButton.disabled = true;
    addPdfBtn.disabled = true;

    welcomeArea.hidden = true;

    // Show the user's question and attached filename.
    const userMessage = fileToUpload
        ? `${question}\n\n📎 ${fileToUpload.name}`
        : question;

    addMessage(userMessage, "user-message");

    questionInput.value = "";

    // Remove the attachment from the composer while processing.
    selectedFile = null;
    chatPdfUpload.value = "";
    renderSelectedPdf();

    const loadingMessage = addMessage(
        fileToUpload ? "Uploading PDF and preparing your answer..." : "Thinking...",
        "assistant-message"
    );

    try {
        // Step 1: Upload and index the PDF, if attached.
        if (fileToUpload) {
            loadingMessage.textContent = `Uploading and indexing ${fileToUpload.name}...`;

            await uploadSelectedPdf(fileToUpload);
        }

        // Step 2: Send the question to the chat endpoint.
        loadingMessage.textContent = "Researching your question...";

        const response = await fetch(CHAT_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || `Chat request failed: ${response.status}`
            );
        }

        loadingMessage.textContent = data.answer || "No answer returned.";

    } catch (error) {
        console.error("Chat workflow failed:", error);

        loadingMessage.textContent =
            `Unable to complete your request: ${error.message}`;

    } finally {
        isSending = false;
        questionInput.disabled = false;
        sendButton.disabled = false;
        addPdfBtn.disabled = false;
        questionInput.focus();
    }
});

// --------------------------------------------------
// New Chat
// --------------------------------------------------

newChatBtn.addEventListener("click", () => {
    if (isSending) {
        return;
    }

    chatMessages.replaceChildren();
    questionInput.value = "";

    selectedFile = null;
    chatPdfUpload.value = "";
    renderSelectedPdf();

    showWelcome();
    showSection(chatSection);

    navItems.forEach(item => item.classList.remove("active"));
});

// --------------------------------------------------
// PDF Library: upload and index immediately
// --------------------------------------------------

document.getElementById("pdf-upload").addEventListener("change", async event => {
    const pdfList = document.getElementById("pdf-list");
    const files = Array.from(event.target.files);

    for (const file of files) {
        if (
            file.type !== "application/pdf" &&
            !file.name.toLowerCase().endsWith(".pdf")
        ) {
            alert(`"${file.name}" is not a PDF file.`);
            continue;
        }

        const card = document.createElement("div");
        card.className = "file-card";

        const name = document.createElement("p");
        name.textContent = file.name;

        const status = document.createElement("p");
        status.textContent = "Uploading PDF and indexing...";
        status.style.marginTop = "8px";

        card.append(name, status);
        pdfList.appendChild(card);

        try {
            const result = await uploadSelectedPdf(file);

            status.textContent =
                `Indexed successfully — ${result.pages} pages, ${result.chunks} chunks.`;

            status.dataset.status = "success";

        } catch (error) {
            console.error("PDF Library upload failed:", error);

            status.textContent = `Upload failed: ${error.message}`;
            status.dataset.status = "error";
        }
    }

    event.target.value = "";
});

// --------------------------------------------------
// Images: local preview only
// --------------------------------------------------

document.getElementById("image-upload").addEventListener("change", event => {
    const imageList = document.getElementById("image-list");

    Array.from(event.target.files).forEach(file => {
        if (!file.type.startsWith("image/")) {
            return;
        }

        const card = document.createElement("div");
        card.className = "file-card";

        const image = document.createElement("img");
        image.src = URL.createObjectURL(file);
        image.alt = file.name;

        const name = document.createElement("p");
        name.textContent = file.name;

        card.append(image, name);
        imageList.appendChild(card);
    });

    event.target.value = "";
});

// --------------------------------------------------
// Suggested questions
// --------------------------------------------------

document.querySelectorAll(".suggestion-chip").forEach(button => {
    button.addEventListener("click", () => {
        questionInput.value = button.dataset.question || "";
        questionInput.focus();
    });
});

// --------------------------------------------------
// Login and logout placeholders
// --------------------------------------------------

document.getElementById("login-btn").addEventListener("click", () => {
    window.location.href = "login.html";
});

document.getElementById("logout-btn").addEventListener("click", () => {
    alert("Authentication is not connected yet.");
});

// --------------------------------------------------
// Initial view
// --------------------------------------------------

showSection(chatSection);
showWelcome();

console.log("AI RESEARCH AGENT INITIALIZED");
