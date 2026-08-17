// 企业文档智能管理系统 — 前端应用
const API = '/api/v1';

// XSS 防护：HTML 实体转义
function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

// ===== Auth =====
const Auth = {
  token: localStorage.getItem('token'),
  user: JSON.parse(localStorage.getItem('user') || 'null'),

  async login() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    const errEl = document.getElementById('login-error');
    if (!username || !password) {
      errEl.textContent = '请输入用户名和密码';
      errEl.classList.remove('hidden');
      return;
    }
    try {
      const r = await fetch(`${API}/auth/login`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || '登录失败');
      Auth.token = data.token;
      Auth.user = data.user;
      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
      document.getElementById('page-login').classList.add('hidden');
      document.getElementById('page-app').classList.remove('hidden');
      document.getElementById('user-info').textContent = `${data.user.username} (${data.user.role})`;
      App.init();
    } catch (e) {
      errEl.textContent = e.message;
      errEl.classList.remove('hidden');
    }
  },

  logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    Auth.token = null;
    Auth.user = null;
    document.getElementById('page-app').classList.add('hidden');
    document.getElementById('page-login').classList.remove('hidden');
  },

  headers() {
    return Auth.token ? { 'Authorization': `Bearer ${Auth.token}`, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
  }
};

// ===== App =====
const App = {
  currentTab: 'dashboard',

  async init() {
    // Sidebar menu
    document.querySelectorAll('.menu-item').forEach(el => {
      el.addEventListener('click', e => {
        e.preventDefault();
        App.switchTab(el.dataset.page);
      });
    });
    App.switchTab('dashboard');
    await App.loadStats();
    await App.loadDocuments();
  },

  switchTab(tab) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.menu-item').forEach(el => el.classList.remove('active'));
    const tabEl = document.getElementById(`tab-${tab}`);
    const menuEl = document.querySelector(`[data-page="${tab}"]`);
    if (tabEl) tabEl.classList.add('active');
    if (menuEl) menuEl.classList.add('active');
    App.currentTab = tab;
    if (tab === 'dashboard') App.loadStats();
    if (tab === 'documents') App.loadDocuments();
    if (tab === 'chat') App.initChat();
  },

  // Stats
  async loadStats() {
    try {
      const r = await fetch(`${API}/stats`, { headers: Auth.headers() });
      if (!r.ok) return;
      const s = await r.json();
      const cards = [
        { label: '文档总数', value: s.documents, color: '#1a73e8' },
        { label: 'Chunk 数', value: s.chunks, color: '#16a34a' },
        { label: 'QA 对数', value: s.qa_pairs, color: '#f59e0b' },
        { label: '处理日志', value: s.audit_log, color: '#8b5cf6' },
      ];
      document.getElementById('stats-cards').innerHTML = cards.map(c =>
        `<div class="stat-card"><div class="stat-value" style="color:${c.color}">${c.value}</div><div class="stat-label">${c.label}</div></div>`
      ).join('');

      // Recent docs
      const rd = await fetch(`${API}/documents`, { headers: Auth.headers() });
      if (rd.ok) {
        const docs = await rd.json();
        document.getElementById('recent-docs').innerHTML = docs.slice(0, 5).map(d =>
          `<div class="result-item"><strong>${d.file_name}</strong> <span class="text-muted">${d.page_count}页 | ${(d.file_size_bytes/1024).toFixed(0)}KB | ${d.created_at}</span></div>`
        ).join('') || '<p class="text-muted">暂无文档</p>';
      }
    } catch (e) { console.error(e); }
  },

  // Documents
  async loadDocuments() {
    try {
      const r = await fetch(`${API}/documents`, { headers: Auth.headers() });
      if (!r.ok) return;
      const docs = await r.json();
      const tbody = document.getElementById('doc-table-body');
      tbody.innerHTML = docs.map(d => `
        <tr>
          <td><strong>${d.file_name}</strong></td>
          <td>${d.page_count}</td>
          <td>${(d.file_size_bytes/1024).toFixed(0)} KB</td>
          <td><span class="badge badge-${d.parse_status==='completed'?'success':'warning'}">${d.parse_status}</span></td>
          <td>${d.created_at || '-'}</td>
          <td><button class="btn btn-sm btn-outline" onclick="App.viewDoc('${d.doc_id}')" style="color:#1f2937;border-color:#d1d5db;">详情</button></td>
        </tr>
      `).join('') || '<tr><td colspan="6" class="text-muted">暂无文档</td></tr>';
    } catch (e) { console.error(e); }
  },

  async viewDoc(docId) {
    try {
      const r = await fetch(`${API}/documents/${docId}`, { headers: Auth.headers() });
      if (!r.ok) return alert('文档不存在');
      const d = await r.json();
      alert(`文档: ${d.file_name}\n页数: ${d.page_count}\nChunks: ${d.chunks?.length || 0}\nQA对: ${d.qa_count || 0}\n标签: ${(d.tags||[]).map(t=>t.name).join(', ') || '无'}`);
    } catch (e) { alert('加载失败'); }
  },

  // Search
  async search() {
    const q = document.getElementById('search-input').value.trim();
    if (!q) return;
    const container = document.getElementById('search-results');
    container.innerHTML = '<p class="text-muted">搜索中...</p>';
    try {
      const r = await fetch(`${API}/search`, {
        method: 'POST', headers: Auth.headers(),
        body: JSON.stringify({ query: q, top_k: 10 })
      });
      if (!r.ok) throw new Error('搜索失败');
      const data = await r.json();
      container.innerHTML = data.results.length === 0
        ? '<p class="text-muted">未找到相关结果</p>'
        : `<p class="text-muted mb-2">找到 ${data.total} 条结果</p>` +
          data.results.map((item, i) => `
            <div class="card mb-2">
              <div class="result-meta">
                <strong>#${i+1}</strong> | ${item.source_file} | 相关度: ${Number(item.score).toFixed(4)}
                ${item.vector_rank ? ` | Vec#${item.vector_rank}` : ''}
                ${item.bm25_rank ? ` | BM25#${item.bm25_rank}` : ''}
              </div>
              <div class="result-content">${item.content}</div>
              ${item.qa_pairs?.length ? `<div class="result-qa">QA: ${item.qa_pairs[0].q} → ${item.qa_pairs[0].a}</div>` : ''}
            </div>
          `).join('');
    } catch (e) {
      container.innerHTML = `<p class="form-error">${escapeHtml(e.message)}</p>`;
    }
  },

  // Chat
  initChat() {
    const container = document.getElementById('chat-messages');
    if (container.children.length === 0) {
      container.innerHTML = '<div class="chat-msg assistant"><div class="chat-bubble">你好！我是文档智能助手。可以帮你搜索知识库、查找文档、回答问题。请随时提问。</div></div>';
    }
  },

  async chat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    const container = document.getElementById('chat-messages');

    // User message (XSS: 转义用户输入)
    container.innerHTML += `<div class="chat-msg user"><div class="chat-bubble">${escapeHtml(msg)}</div></div>`;
    input.value = '';

    // Assistant placeholder
    const placeholder = document.createElement('div');
    placeholder.className = 'chat-msg assistant';
    placeholder.innerHTML = '<div class="chat-bubble">思考中...</div>';
    container.appendChild(placeholder);
    container.scrollTop = container.scrollHeight;

    try {
      const r = await fetch(`${API}/chat`, {
        method: 'POST', headers: Auth.headers(),
        body: JSON.stringify({ message: msg })
      });
      if (!r.ok) throw new Error('对话失败');
      const data = await r.json();
      // XSS: 逐行转义后保留换行
      const safeAnswer = data.answer.split('\n').map(line => escapeHtml(line)).join('<br>');
      placeholder.innerHTML = `<div class="chat-bubble">${safeAnswer}</div>`;
    } catch (e) {
      placeholder.innerHTML = `<div class="chat-bubble" style="color:var(--danger)">错误: ${escapeHtml(e.message)}</div>`;
    }
    container.scrollTop = container.scrollHeight;
  },

  async resetChat() {
    try {
      await fetch(`${API}/chat`, {
        method: 'POST', headers: Auth.headers(),
        body: JSON.stringify({ message: 'reset', reset: true })
      });
      document.getElementById('chat-messages').innerHTML = '';
      App.initChat();
    } catch (e) { /* ignore */ }
  },

  // Upload (supports multiple files)
  async uploadFiles(input) {
    const files = Array.from(input.files);
    if (!files.length) return;
    const status = document.getElementById('upload-status');
    status.innerHTML = '';

    let ok = 0, fail = 0;
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const item = document.createElement('p');
      item.textContent = `[${i+1}/${files.length}] ${file.name} — 处理中...`;
      status.appendChild(item);

      try {
        const formData = new FormData();
        formData.append('file', file);
        const r = await fetch(`${API}/documents/upload`, {
          method: 'POST', headers: { 'Authorization': `Bearer ${Auth.token}` },
          body: formData
        });
        if (!r.ok) {
          const err = await r.json();
          throw new Error(err.detail || '上传失败');
        }
        const d = await r.json();
        item.innerHTML = `[${i+1}/${files.length}] <span style="color:var(--success)">${file.name}</span> — ${d.chunk_count} chunks`;
        ok++;
      } catch (e) {
        item.innerHTML = `[${i+1}/${files.length}] <span style="color:var(--danger)">${file.name}</span> — ${escapeHtml(e.message)}`;
        fail++;
      }
    }

    const summary = document.createElement('p');
    summary.style.marginTop = '8px';
    summary.style.fontWeight = 'bold';
    summary.textContent = `完成: ${ok} 成功, ${fail} 失败`;
    status.appendChild(summary);

    input.value = '';
    App.loadStats();
    App.loadDocuments();
  }
};

// ===== Init =====
(function() {
  // Drag & drop for upload zone
  document.addEventListener('DOMContentLoaded', () => {
    const zone = document.getElementById('upload-zone');
    if (zone) {
      zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--primary)'; });
      zone.addEventListener('dragleave', () => { zone.style.borderColor = 'var(--border)'; });
      zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.style.borderColor = 'var(--border)';
        const pdfFiles = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.pdf'));
        if (pdfFiles.length) {
          const input = document.getElementById('file-input');
          const dt = new DataTransfer();
          pdfFiles.forEach(f => dt.items.add(f));
          input.files = dt.files;
          App.uploadFiles(input);
        }
      });
    }

    // Login form submit
    document.getElementById('login-form').addEventListener('submit', e => {
      e.preventDefault();
      Auth.login();
    });

    // Auto login if token exists
    if (Auth.token && Auth.user) {
      document.getElementById('page-login').classList.add('hidden');
      document.getElementById('page-app').classList.remove('hidden');
      document.getElementById('user-info').textContent = `${Auth.user.username} (${Auth.user.role})`;
      App.init();
    }
  });
})();
