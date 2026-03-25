(function () {
    "use strict";

    var listEl = document.getElementById("session-list");
    var paginationEl = document.getElementById("pagination");
    var searchInput = document.getElementById("search-input");
    var searchBtn = document.getElementById("search-btn");
    var modal = document.getElementById("detail-modal");
    var modalClose = document.getElementById("modal-close");
    var detailHeader = document.getElementById("detail-header");
    var audioContainer = document.getElementById("audio-player-container");
    var segmentsContainer = document.getElementById("segments-container");

    // Settings elements
    var settingsBtn = document.getElementById("settings-btn");
    var settingsModal = document.getElementById("settings-modal");
    var settingsClose = document.getElementById("settings-close");

    var currentPage = 1;
    var currentQuery = "";
    var audioEl = null;

    // In-memory caches
    var termsCache = null;
    var autoDetectedCache = [];

    // --- Helpers ---

    function formatDuration(sec) {
        if (!sec && sec !== 0) return "-";
        var m = Math.floor(sec / 60);
        var s = Math.round(sec % 60);
        return m + ":" + (s < 10 ? "0" : "") + s;
    }

    function formatTimestamp(sec) {
        var m = Math.floor(sec / 60);
        var s = Math.floor(sec % 60);
        return m + ":" + (s < 10 ? "0" : "") + s;
    }

    function formatDateTime(date, time) {
        var t = time.replace(/-/g, ":");
        return date + " " + t;
    }

    function escapeHtml(text) {
        var d = document.createElement("div");
        d.textContent = text;
        return d.innerHTML;
    }

    // --- API ---

    function fetchSessions(page, query) {
        var url = "/api/sessions?page=" + page + "&per_page=20";
        if (query) url += "&q=" + encodeURIComponent(query);
        return fetch(url).then(function (r) { return r.json(); });
    }

    function fetchSessionDetail(date, time) {
        return fetch("/api/sessions/" + date + "/" + time).then(function (r) { return r.json(); });
    }

    // --- Render ---

    function renderSessions(data) {
        if (!data.sessions || data.sessions.length === 0) {
            listEl.innerHTML = '<div class="empty-state">' +
                (currentQuery ? "No results for: " + escapeHtml(currentQuery) : "No recordings found") +
                "</div>";
            paginationEl.innerHTML = "";
            return;
        }

        var html = "";
        data.sessions.forEach(function (s) {
            var preview = s.transcript_text || "";
            if (preview.length > 150) preview = preview.substring(0, 150) + "...";
            var isEmpty = !preview;

            html += '<div class="session-card" data-date="' + s.date + '" data-time="' + s.time + '">';
            html += '<div class="session-meta">';
            html += '<span class="date">' + formatDateTime(s.date, s.time) + "</span>";
            html += "<span>" + formatDuration(s.duration_seconds) + "</span>";
            if (s.language) html += "<span>" + escapeHtml(s.language) + "</span>";
            if (s.segment_count) html += "<span>" + s.segment_count + " segments</span>";
            if (s.audio_size_mb) html += "<span>" + s.audio_size_mb.toFixed(1) + " MB</span>";
            html += "</div>";
            html += '<div class="session-preview' + (isEmpty ? " empty" : "") + '">';
            html += isEmpty ? "(no transcript)" : escapeHtml(preview);
            html += "</div>";
            html += "</div>";
        });

        listEl.innerHTML = html;
        renderPagination(data);

        var cards = listEl.querySelectorAll(".session-card");
        for (var i = 0; i < cards.length; i++) {
            cards[i].addEventListener("click", onCardClick);
        }
    }

    function renderPagination(data) {
        if (data.pages <= 1) {
            paginationEl.innerHTML = "";
            return;
        }

        var html = "";
        html += '<button ' + (data.page <= 1 ? "disabled" : "") + ' data-page="' + (data.page - 1) + '">&laquo;</button>';

        var start = Math.max(1, data.page - 2);
        var end = Math.min(data.pages, data.page + 2);

        if (start > 1) {
            html += '<button data-page="1">1</button>';
            if (start > 2) html += '<span class="page-info">...</span>';
        }

        for (var p = start; p <= end; p++) {
            html += '<button data-page="' + p + '"' + (p === data.page ? ' class="active"' : "") + ">" + p + "</button>";
        }

        if (end < data.pages) {
            if (end < data.pages - 1) html += '<span class="page-info">...</span>';
            html += '<button data-page="' + data.pages + '">' + data.pages + "</button>";
        }

        html += '<button ' + (data.page >= data.pages ? "disabled" : "") + ' data-page="' + (data.page + 1) + '">&raquo;</button>';
        html += '<span class="page-info">' + data.total + " sessions</span>";

        paginationEl.innerHTML = html;

        var buttons = paginationEl.querySelectorAll("button[data-page]");
        for (var i = 0; i < buttons.length; i++) {
            buttons[i].addEventListener("click", function () {
                var p = parseInt(this.getAttribute("data-page"), 10);
                if (p >= 1) loadPage(p);
            });
        }
    }

    function renderDetail(data) {
        var s = data.session;
        var t = data.transcript;

        detailHeader.innerHTML =
            "<h2>" + formatDateTime(s.date, s.time) + "</h2>" +
            '<div class="meta">' +
            "<span>Duration: " + formatDuration(s.duration_seconds) + "</span>" +
            (s.language ? "<span>Language: " + escapeHtml(s.language) + "</span>" : "") +
            (s.model ? "<span>Model: " + escapeHtml(s.model) + "</span>" : "") +
            (s.audio_size_mb ? "<span>Size: " + s.audio_size_mb.toFixed(1) + " MB</span>" : "") +
            "</div>";

        var audioFile = (t && t.session && t.session.audio_file) ? t.session.audio_file : "audio.ogg";
        audioContainer.innerHTML =
            '<audio id="audio-el" controls preload="metadata" src="/audio/' + s.date + "/" + s.time + '/' + audioFile + '"></audio>';
        audioEl = document.getElementById("audio-el");

        audioEl.addEventListener("timeupdate", onTimeUpdate);

        var segments = (t && t.segments) ? t.segments : [];
        if (segments.length === 0) {
            segmentsContainer.innerHTML = '<div class="loading">No segments</div>';
            return;
        }

        var html = "";
        segments.forEach(function (seg, idx) {
            html += '<div class="segment" data-idx="' + idx + '" data-start="' + seg.start + '" data-end="' + seg.end + '">';
            html += '<span class="segment-time" data-seek="' + seg.start + '">' + formatTimestamp(seg.start) + "</span>";
            html += '<span class="segment-speaker">' + (seg.speaker || "") + "</span>";
            html += '<span class="segment-text">' + escapeHtml(seg.text) + "</span>";
            html += "</div>";
        });
        segmentsContainer.innerHTML = html;

        var times = segmentsContainer.querySelectorAll(".segment-time");
        for (var i = 0; i < times.length; i++) {
            times[i].addEventListener("click", function (e) {
                e.stopPropagation();
                var seek = parseFloat(this.getAttribute("data-seek"));
                if (audioEl) {
                    audioEl.currentTime = seek;
                    audioEl.play();
                }
            });
        }
    }

    function onTimeUpdate() {
        if (!audioEl) return;
        var ct = audioEl.currentTime;
        var segs = segmentsContainer.querySelectorAll(".segment");
        for (var i = 0; i < segs.length; i++) {
            var start = parseFloat(segs[i].getAttribute("data-start"));
            var end = parseFloat(segs[i].getAttribute("data-end"));
            if (ct >= start && ct < end) {
                segs[i].classList.add("active");
            } else {
                segs[i].classList.remove("active");
            }
        }
    }

    // --- Events ---

    function loadPage(page) {
        currentPage = page;
        listEl.innerHTML = '<div class="loading">Loading...</div>';
        fetchSessions(page, currentQuery).then(renderSessions);
    }

    function doSearch() {
        currentQuery = searchInput.value.trim();
        currentPage = 1;
        loadPage(1);
    }

    function onCardClick() {
        var date = this.getAttribute("data-date");
        var time = this.getAttribute("data-time");
        openDetail(date, time);
    }

    function openDetail(date, time) {
        detailHeader.innerHTML = '<div class="loading">Loading...</div>';
        audioContainer.innerHTML = "";
        segmentsContainer.innerHTML = "";
        modal.classList.remove("hidden");

        fetchSessionDetail(date, time).then(renderDetail);
    }

    function closeModal() {
        modal.classList.add("hidden");
        if (audioEl) {
            audioEl.pause();
            audioEl = null;
        }
    }

    searchBtn.addEventListener("click", doSearch);
    searchInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") doSearch();
    });
    modalClose.addEventListener("click", closeModal);
    modal.addEventListener("click", function (e) {
        if (e.target === modal) closeModal();
    });

    // ============================================================
    // Settings panel
    // ============================================================

    function openSettings() {
        settingsModal.classList.remove("hidden");
        loadPromptSettings();
        loadTerms();
        loadAutoDetected();
        loadSuggestions();
        loadSuggestionCount();
        loadPresets();
    }

    function closeSettings() {
        settingsModal.classList.add("hidden");
    }

    settingsBtn.addEventListener("click", openSettings);
    settingsClose.addEventListener("click", closeSettings);
    settingsModal.addEventListener("click", function (e) {
        if (e.target === settingsModal) closeSettings();
    });

    // Tabs
    var tabs = document.querySelectorAll(".tab");
    for (var i = 0; i < tabs.length; i++) {
        tabs[i].addEventListener("click", function () {
            var targetId = this.getAttribute("data-tab");
            document.querySelectorAll(".tab").forEach(function (t) { t.classList.remove("active"); });
            document.querySelectorAll(".tab-content").forEach(function (c) { c.classList.remove("active"); });
            this.classList.add("active");
            document.getElementById(targetId).classList.add("active");
        });
    }

    // --- Prompt settings ---

    var promptTopic = document.getElementById("prompt-topic");
    var promptParticipants = document.getElementById("prompt-participants");
    var promptTerms = document.getElementById("prompt-terms");
    var promptPreview = document.getElementById("prompt-preview");
    var promptCharCount = document.getElementById("prompt-char-count");
    var promptWarnings = document.getElementById("prompt-warnings");
    var savePromptBtn = document.getElementById("save-prompt");
    var promptStatus = document.getElementById("prompt-status");

    function buildPreview() {
        var topic = promptTopic.value.trim();
        var parts = [];
        if (topic) parts.push(topic + "に関する会議です。");

        var pp = promptParticipants.value.trim();
        if (pp) {
            var names = pp.split(/[,、]/).map(function (s) { return s.trim(); }).filter(Boolean);
            if (names.length) parts.push("参加者: " + names.join("、") + "。");
        }

        var tt = promptTerms.value.trim();
        if (tt) {
            var terms = tt.split(/[,、]/).map(function (s) { return s.trim(); }).filter(Boolean);
            if (terms.length) {
                var selected = terms.slice(0, 15);
                parts.push("関連用語: " + selected.join("、") + "。");
            }
        }

        var text = parts.length ? parts.join("") : "";

        // Update char count
        if (text) {
            var len = text.length;
            promptCharCount.textContent = len + " / 180文字";
            if (len > 180) {
                promptCharCount.style.color = "#e74c3c";
            } else {
                promptCharCount.style.color = "";
            }
        } else {
            promptCharCount.textContent = "";
        }

        // Warnings
        var warnings = [];
        if (text && text.length > 180) {
            warnings.push("文字数が180文字を超えています。Whisperの精度が低下する可能性があります。");
        }
        if (text && /[。.]$/.test(text)) {
            warnings.push("末尾が句点で終わっています。Whisperが句点をエコーする場合があります。");
        }
        if (warnings.length > 0) {
            promptWarnings.innerHTML = warnings.map(function (w) {
                return '<span class="warning-text">' + escapeHtml(w) + '</span>';
            }).join("");
        } else {
            promptWarnings.innerHTML = "";
        }

        if (text) {
            if (text.length > 180) text = text.substring(0, 180) + "...";
            promptPreview.textContent = text;
        } else {
            promptPreview.textContent = "プロンプトなし";
        }
    }

    promptTopic.addEventListener("input", buildPreview);
    promptParticipants.addEventListener("input", buildPreview);
    promptTerms.addEventListener("input", buildPreview);

    function loadPromptSettings() {
        fetch("/api/prompt-settings").then(function (r) { return r.json(); }).then(function (data) {
            promptTopic.value = data.topic || "";
            promptParticipants.value = (data.participants || []).join(", ");
            promptTerms.value = (data.terms || []).join(", ");
            buildPreview();
        });
    }

    savePromptBtn.addEventListener("click", function () {
        var pp = promptParticipants.value.trim();
        var participants = pp ? pp.split(/[,、]/).map(function (s) { return s.trim(); }).filter(Boolean) : [];

        var tt = promptTerms.value.trim();
        var terms = tt ? tt.split(/[,、]/).map(function (s) { return s.trim(); }).filter(Boolean) : [];

        fetch("/api/prompt-settings", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                topic: promptTopic.value.trim(),
                participants: participants,
                terms: terms,
            }),
        }).then(function (r) { return r.json(); }).then(function () {
            promptStatus.textContent = "Saved!";
            setTimeout(function () { promptStatus.textContent = ""; }, 2000);
        });
    });

    // --- Term corrections ---

    var termWrongInput = document.getElementById("term-wrong");
    var termCorrectInput = document.getElementById("term-correct");
    var termAddBtn = document.getElementById("term-add-btn");
    var termFilterInput = document.getElementById("term-filter");
    var termCountEl = document.getElementById("term-count");
    var termsListEl = document.getElementById("terms-list");

    function loadTerms() {
        fetch("/api/terms").then(function (r) { return r.json(); }).then(function (data) {
            termsCache = data;
            renderTerms(data);
        });
    }

    function loadAutoDetected() {
        fetch("/api/terms/auto-detected").then(function (r) { return r.json(); }).then(function (data) {
            autoDetectedCache = data || [];
        });
    }

    function renderTerms(data, filter) {
        var keys = Object.keys(data).sort(function (a, b) {
            var va = data[a].toLowerCase(), vb = data[b].toLowerCase();
            if (va < vb) return -1;
            if (va > vb) return 1;
            return a.localeCompare(b);
        });

        if (filter) {
            var f = filter.toLowerCase();
            keys = keys.filter(function (k) {
                return k.toLowerCase().indexOf(f) !== -1 || data[k].toLowerCase().indexOf(f) !== -1;
            });
        }

        termCountEl.textContent = keys.length + " / " + Object.keys(data).length + " entries";

        if (keys.length === 0) {
            termsListEl.innerHTML = '<div class="loading">No entries</div>';
            return;
        }

        var html = "";
        keys.forEach(function (wrong) {
            var isAuto = autoDetectedCache.indexOf(wrong) !== -1;
            html += '<div class="term-row" data-wrong="' + escapeHtml(wrong) + '">';
            html += '<span class="term-wrong">' + escapeHtml(wrong) + '</span>';
            if (isAuto) html += '<span class="badge-auto">auto</span>';
            html += '<span class="term-arrow">&rarr;</span>';
            html += '<span class="term-correct" data-wrong="' + escapeHtml(wrong) + '">' + escapeHtml(data[wrong]) + '</span>';
            html += '<button class="btn-edit term-edit" data-wrong="' + escapeHtml(wrong) + '">Edit</button>';
            html += '<button class="btn-danger term-delete" data-wrong="' + escapeHtml(wrong) + '">Del</button>';
            html += "</div>";
        });
        termsListEl.innerHTML = html;

        var delBtns = termsListEl.querySelectorAll(".term-delete");
        for (var i = 0; i < delBtns.length; i++) {
            delBtns[i].addEventListener("click", function () {
                var wrong = this.getAttribute("data-wrong");
                deleteTerm(wrong);
            });
        }

        var editBtns = termsListEl.querySelectorAll(".term-edit");
        for (var j = 0; j < editBtns.length; j++) {
            editBtns[j].addEventListener("click", function () {
                var wrong = this.getAttribute("data-wrong");
                editTermInline(wrong);
            });
        }
    }

    function editTermInline(wrong) {
        var correctSpan = termsListEl.querySelector('.term-correct[data-wrong="' + wrong + '"]');
        if (!correctSpan || correctSpan.querySelector("input")) return;

        var currentValue = termsCache[wrong] || "";
        var input = document.createElement("input");
        input.type = "text";
        input.className = "edit-input";
        input.value = currentValue;
        correctSpan.textContent = "";
        correctSpan.appendChild(input);
        input.focus();

        function save() {
            var newCorrect = input.value.trim();
            if (!newCorrect || newCorrect === currentValue) {
                correctSpan.textContent = currentValue;
                return;
            }
            fetch("/api/terms/" + encodeURIComponent(wrong), {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ wrong: wrong, correct: newCorrect }),
            }).then(function (r) { return r.json(); }).then(function () {
                termsCache = null;
                loadTerms();
            });
        }

        input.addEventListener("keydown", function (e) {
            if (e.key === "Enter") save();
            if (e.key === "Escape") correctSpan.textContent = currentValue;
        });
        input.addEventListener("blur", save);
    }

    termFilterInput.addEventListener("input", function () {
        if (termsCache) renderTerms(termsCache, this.value.trim());
    });

    termAddBtn.addEventListener("click", addTerm);
    termCorrectInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") addTerm();
    });

    function addTerm() {
        var wrong = termWrongInput.value.trim();
        var correct = termCorrectInput.value.trim();
        if (!wrong || !correct) return;

        fetch("/api/terms", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ wrong: wrong, correct: correct }),
        }).then(function (r) { return r.json(); }).then(function () {
            termWrongInput.value = "";
            termCorrectInput.value = "";
            loadTerms();
        });
    }

    function deleteTerm(wrong) {
        fetch("/api/terms", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ wrong: wrong }),
        }).then(function (r) { return r.json(); }).then(function () {
            loadTerms();
        });
    }

    // ============================================================
    // LLM Suggestions
    // ============================================================

    var suggestionsListEl = document.getElementById("suggestions-list");
    var suggestionBadge = document.getElementById("suggestion-badge");

    function loadSuggestionCount() {
        fetch("/api/suggestions").then(function (r) { return r.json(); }).then(function (data) {
            var count = data.length;
            if (count > 0) {
                suggestionBadge.textContent = count;
                suggestionBadge.classList.remove("hidden");
            } else {
                suggestionBadge.classList.add("hidden");
            }
        });
    }

    function loadSuggestions() {
        fetch("/api/suggestions").then(function (r) { return r.json(); }).then(function (data) {
            renderSuggestions(data);
            // Update badge
            if (data.length > 0) {
                suggestionBadge.textContent = data.length;
                suggestionBadge.classList.remove("hidden");
            } else {
                suggestionBadge.classList.add("hidden");
            }
        });
    }

    function renderSuggestions(list) {
        if (list.length === 0) {
            suggestionsListEl.innerHTML = '<div class="loading">No pending suggestions</div>';
            return;
        }

        var html = "";
        list.forEach(function (s) {
            html += '<div class="suggestion-row" data-id="' + escapeHtml(s.id) + '">';
            html += '<div class="suggestion-diff">';
            html += '<span class="wrong">' + escapeHtml(s.wrong) + '</span>';
            html += ' &rarr; ';
            html += '<span class="correct" data-id="' + escapeHtml(s.id) + '">' + escapeHtml(s.correct) + '</span>';
            html += '</div>';
            if (s.context) {
                html += '<div class="suggestion-context">' + escapeHtml(s.context) + '</div>';
            }
            html += '<div class="suggestion-actions">';
            html += '<button class="btn-sm suggestion-approve" data-id="' + escapeHtml(s.id) + '">Approve</button>';
            html += '<button class="btn-edit suggestion-edit-btn" data-id="' + escapeHtml(s.id) + '">Edit</button>';
            html += '<button class="btn-danger suggestion-reject" data-id="' + escapeHtml(s.id) + '">Reject</button>';
            html += '</div>';
            html += '</div>';
        });
        suggestionsListEl.innerHTML = html;

        // Bind events
        var approveBtns = suggestionsListEl.querySelectorAll(".suggestion-approve");
        for (var i = 0; i < approveBtns.length; i++) {
            approveBtns[i].addEventListener("click", function () {
                approveSuggestion(this.getAttribute("data-id"));
            });
        }

        var rejectBtns = suggestionsListEl.querySelectorAll(".suggestion-reject");
        for (var j = 0; j < rejectBtns.length; j++) {
            rejectBtns[j].addEventListener("click", function () {
                rejectSuggestion(this.getAttribute("data-id"));
            });
        }

        var editBtns = suggestionsListEl.querySelectorAll(".suggestion-edit-btn");
        for (var k = 0; k < editBtns.length; k++) {
            editBtns[k].addEventListener("click", function () {
                editSuggestion(this.getAttribute("data-id"));
            });
        }
    }

    function approveSuggestion(id) {
        fetch("/api/suggestions/" + id + "/approve", {
            method: "POST",
        }).then(function (r) { return r.json(); }).then(function () {
            termsCache = null;
            loadSuggestions();
            loadTerms();
            loadAutoDetected();
        });
    }

    function rejectSuggestion(id) {
        fetch("/api/suggestions/" + id + "/reject", {
            method: "POST",
        }).then(function (r) { return r.json(); }).then(function () {
            loadSuggestions();
        });
    }

    function editSuggestion(id) {
        var correctSpan = suggestionsListEl.querySelector('.correct[data-id="' + id + '"]');
        if (!correctSpan || correctSpan.querySelector("input")) return;

        var currentValue = correctSpan.textContent;
        var input = document.createElement("input");
        input.type = "text";
        input.className = "suggestion-edit-input";
        input.value = currentValue;
        correctSpan.textContent = "";
        correctSpan.appendChild(input);
        input.focus();

        function save() {
            var newCorrect = input.value.trim();
            if (!newCorrect || newCorrect === currentValue) {
                correctSpan.textContent = currentValue;
                return;
            }
            fetch("/api/suggestions/" + id, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ correct: newCorrect }),
            }).then(function (r) { return r.json(); }).then(function () {
                loadSuggestions();
            });
        }

        input.addEventListener("keydown", function (e) {
            if (e.key === "Enter") save();
            if (e.key === "Escape") correctSpan.textContent = currentValue;
        });
        input.addEventListener("blur", save);
    }

    // ============================================================
    // Presets
    // ============================================================

    var presetSelect = document.getElementById("preset-select");
    var presetSaveBtn = document.getElementById("preset-save-btn");
    var presetDeleteBtn = document.getElementById("preset-delete-btn");

    function loadPresets() {
        fetch("/api/presets").then(function (r) { return r.json(); }).then(function (presets) {
            // Preserve current selection if possible
            var currentVal = presetSelect.value;
            presetSelect.innerHTML = '<option value="">-- Select Preset --</option>';
            presets.forEach(function (p) {
                var opt = document.createElement("option");
                opt.value = p.id;
                opt.textContent = p.name;
                opt.dataset.topic = p.topic || "";
                opt.dataset.participants = (p.participants || []).join(", ");
                opt.dataset.terms = (p.terms || []).join(", ");
                presetSelect.appendChild(opt);
            });
            // Restore selection
            if (currentVal) {
                presetSelect.value = currentVal;
            }
        });
    }

    presetSelect.addEventListener("change", function () {
        var opt = presetSelect.options[presetSelect.selectedIndex];
        if (!opt || !opt.value) return;

        promptTopic.value = opt.dataset.topic || "";
        promptParticipants.value = opt.dataset.participants || "";
        promptTerms.value = opt.dataset.terms || "";
        buildPreview();
    });

    presetSaveBtn.addEventListener("click", function () {
        var name = prompt("Preset name:");
        if (!name || !name.trim()) return;

        var pp = promptParticipants.value.trim();
        var participants = pp ? pp.split(/[,、]/).map(function (s) { return s.trim(); }).filter(Boolean) : [];

        var tt = promptTerms.value.trim();
        var terms = tt ? tt.split(/[,、]/).map(function (s) { return s.trim(); }).filter(Boolean) : [];

        fetch("/api/presets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: name.trim(),
                topic: promptTopic.value.trim(),
                participants: participants,
                terms: terms,
            }),
        }).then(function (r) { return r.json(); }).then(function (preset) {
            loadPresets();
            // Select the newly created preset
            setTimeout(function () {
                presetSelect.value = preset.id;
            }, 200);
        });
    });

    presetDeleteBtn.addEventListener("click", function () {
        var pid = presetSelect.value;
        if (!pid) return;
        var name = presetSelect.options[presetSelect.selectedIndex].textContent;
        if (!confirm('Delete preset "' + name + '"?')) return;

        fetch("/api/presets/" + pid, {
            method: "DELETE",
        }).then(function (r) { return r.json(); }).then(function () {
            presetSelect.value = "";
            loadPresets();
        });
    });

    // Escape key closes whichever modal is open
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") {
            if (!settingsModal.classList.contains("hidden")) {
                closeSettings();
            } else {
                closeModal();
            }
        }
    });

    // --- Init ---
    loadPage(1);
})();
