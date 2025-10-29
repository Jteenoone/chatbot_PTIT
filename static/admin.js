// Simple toast
function toast(msg, ok=true){
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "fixed bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-xl shadow " + (ok ? "bg-neutral-900 text-white" : "bg-red-600 text-white");
  t.style.display = "block";
  setTimeout(()=>{ t.style.display="none"; }, 2200);
}

// DOM
const authBox = document.getElementById("authBox");
const tools = document.getElementById("tools");
const btnAuth = document.getElementById("btnAuth");
const adminPass = document.getElementById("adminPass");
const authState = document.getElementById("authState");

const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const pwUpload = document.getElementById("pwUpload");
const uploadMsg = document.getElementById("uploadMsg");

const resetForm = document.getElementById("resetForm");
const pwReset = document.getElementById("pwReset");
const resetMsg = document.getElementById("resetMsg");

// ADM1: check password via API
btnAuth.addEventListener("click", async ()=>{
  const pw = (adminPass.value || "").trim();
  if(!pw){ toast("Nhập mật khẩu", false); return; }

  try{
    const res = await fetch("/check-admin-password", {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ password: pw })
    });
    if(res.ok){
      const data = await res.json();
      if(data.success){
        authState.textContent = "Xác thực thành công.";
        authBox.classList.add("hidden");
        tools.classList.remove("hidden");
        toast("Đăng nhập admin thành công");
        // Autofill password inputs for convenience
        pwUpload.value = pw;
        pwReset.value = pw;
      } else {
        toast("Sai mật khẩu", false);
      }
    } else {
      toast("Sai mật khẩu", false);
    }
  }catch{
    toast("Lỗi kết nối server", false);
  }
});

// Upload + auto update KB
uploadForm.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const f = fileInput.files[0];
  const pw = (pwUpload.value || "").trim();
  if(!f){ toast("Chọn file trước", false); return; }
  if(!pw){ toast("Thiếu mật khẩu", false); return; }

  const form = new FormData();
  form.append("file", f);
  form.append("password", pw);

  uploadMsg.textContent = "Đang tải lên và cập nhật KB…";
  try{
    const res = await fetch("/upload", { method: "POST", body: form });
    const data = await res.json();
    if(res.ok && data.success){
      uploadMsg.textContent = "Đã cập nhật KB thành công.";
      toast("Cập nhật KB thành công");
      // Gợi ý: reload để cập nhật danh sách file
      setTimeout(()=>location.reload(), 800);
    }else{
      uploadMsg.textContent = data.error || "Lỗi không xác định.";
      toast(uploadMsg.textContent, false);
    }
  }catch{
    uploadMsg.textContent = "Lỗi kết nối server.";
    toast(uploadMsg.textContent, false);
  }
});

// Reset KB
resetForm.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const pw = (pwReset.value || "").trim();
  if(!pw){ toast("Thiếu mật khẩu", false); return; }

  const form = new FormData();
  form.append("password", pw);
  resetMsg.textContent = "Đang reset KB…";
  try{
    const res = await fetch("/reset-knowledge", { method: "POST", body: form });
    const data = await res.json();
    if(res.ok && data.success){
      resetMsg.textContent = "Reset xong, đã nạp lại tri thức.";
      toast("Reset thành công");
    }else{
      resetMsg.textContent = data.error || "Reset thất bại.";
      toast(resetMsg.textContent, false);
    }
  }catch{
    resetMsg.textContent = "Lỗi kết nối server.";
    toast(resetMsg.textContent, false);
  }
});
