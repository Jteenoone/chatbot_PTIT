const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

function addMsg(role, text){
  const wrap = document.createElement("div");
  wrap.className = role === "user"
    ? "flex items-start gap-3 justify-end"
    : "flex items-start gap-3";

  const avatar = document.createElement("div");
  avatar.className = "w-9 h-9 rounded-full grid place-items-center text-white " + (role==="user" ? "bg-gray-900" : "bg-red-600");
  avatar.textContent = role === "user" ? "👤" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = role === "user"
    ? "max-w-[80%] rounded-2xl px-4 py-3 shadow bg-red-600 text-white"
    : "max-w-[80%] rounded-2xl px-4 py-3 shadow border border-red-100 bg-white whitespace-pre-wrap";

  bubble.textContent = text;

  if(role==="user"){
    wrap.appendChild(bubble);
    wrap.appendChild(avatar);
  } else {
    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
  }

  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function addTyping(){
  const wrap = document.createElement("div");
  wrap.className = "flex items-start gap-3";
  const avatar = document.createElement("div");
  avatar.className = "w-9 h-9 rounded-full bg-red-600 text-white grid place-items-center";
  avatar.textContent = "🤖";
  const bubble = document.createElement("div");
  bubble.className = "max-w-[80%] rounded-2xl px-4 py-3 shadow border border-red-100 bg-white";
  bubble.innerHTML = `<span class="inline-flex items-center gap-1 text-sm">Đang trả lời<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></span>`;
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

async function send(){
  const q = (inputEl.value || "").trim();
  if(!q) return;
  inputEl.value = "";

  addMsg("user", q);
  const typingEl = addTyping();
  sendBtn.disabled = true;

  try{
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ message: q })
    });
    const data = await res.json();
    typingEl.remove();
    if(data.error){ addMsg("bot", `Lỗi: ${data.error}`); }
    else { addMsg("bot", data.reply || "(Không có trả lời)"); }
  }catch(e){
    typingEl.remove();
    addMsg("bot", "(Lỗi kết nối server)");
  }

  sendBtn.disabled = false;
  inputEl.focus();
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e)=>{
  if(e.key === "Enter" && !e.shiftKey){
    e.preventDefault();
    send();
  }
});
