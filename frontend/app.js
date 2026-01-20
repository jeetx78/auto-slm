const chat = document.getElementById("chat");
const promptInput = document.getElementById("prompt");

const API_URL = "http://127.0.0.1:8000/infer";
// if using ngrok → replace with ngrok URL

function addMessage(text, cls) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.innerText = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  addMessage("You: " + prompt, "user");
  promptInput.value = "";

  addMessage("Auto-SLM is thinking…", "bot");

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt })
    });

    const data = await res.json();
    chat.lastChild.remove();

    if (data.success) {
      addMessage("Auto-SLM: " + data.response, "bot");
    } else {
      addMessage("Error: " + data.error, "bot");
    }
  } catch (err) {
    chat.lastChild.remove();
    addMessage("Network error", "bot");
  }
}
