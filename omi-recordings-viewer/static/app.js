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

    // Speaker / voiceprint state
    var currentDetailDate = "";
    var currentDetailTime = "";
    var currentDetailSegments = [];
    var speakersCache = [];
    var segmentFilter = "all";

    // User tags
    var userTagsCache = [];
    var currentSessionUserTags = [];  // tag IDs for current detail session
    var currentSessionAutoUserTags = [];  // auto-assigned tag IDs (from LLM)

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

    var filterActivity = document.getElementById("filter-activity");
    var filterMood = document.getElementById("filter-mood");
    var filterUserTag = document.getElementById("filter-user-tag");

    function fetchSessions(page, query) {
        var url = "/api/sessions?page=" + page + "&per_page=20";
        if (query) url += "&q=" + encodeURIComponent(query);
        if (filterActivity && filterActivity.value) url += "&activity=" + encodeURIComponent(filterActivity.value);
        if (filterMood && filterMood.value) url += "&mood=" + encodeURIComponent(filterMood.value);
        if (filterUserTag && filterUserTag.value) url += "&user_tag=" + encodeURIComponent(filterUserTag.value);
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
            // Tag badges
            if (s.tags_activity || s.tags_mood || s.tags_topics) {
                html += '<div class="session-tags">';
                if (s.tags_activity) {
                    html += '<span class="tag-badge activity">' + escapeHtml(s.tags_activity.replace(/_/g, " ")) + '</span>';
                }
                if (s.tags_mood) {
                    html += '<span class="tag-badge mood-' + s.tags_mood + '">' + escapeHtml(s.tags_mood) + '</span>';
                }
                if (s.tags_topics) {
                    var topics = s.tags_topics.split(",");
                    for (var ti = 0; ti < topics.length && ti < 3; ti++) {
                        html += '<span class="tag-badge topic">' + escapeHtml(topics[ti].trim()) + '</span>';
                    }
                }
                html += '</div>';
            }
            // User tag badges
            if (s.tags_user && userTagsCache.length > 0) {
                var uTagIds = s.tags_user.split(",");
                var uTagHtml = '';
                for (var ui = 0; ui < uTagIds.length; ui++) {
                    var utId = uTagIds[ui].trim();
                    var utDef = userTagsCache.find(function(t) { return t.id === utId; });
                    if (utDef) {
                        uTagHtml += '<span class="user-tag-card-badge" style="background:' + escapeHtml(utDef.color) + '">' + escapeHtml(utDef.name) + '</span>';
                    }
                }
                if (uTagHtml) {
                    if (!s.tags_activity && !s.tags_mood && !s.tags_topics) {
                        html += '<div class="session-tags">' + uTagHtml + '</div>';
                    } else {
                        // Append to existing tags div — we need to inject before closing
                        // Actually since we already closed the div above, just add another row
                        html += '<div class="session-tags">' + uTagHtml + '</div>';
                    }
                }
            }
            if (s.tags_summary) {
                html += '<div class="session-summary">' + escapeHtml(s.tags_summary) + '</div>';
            }
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

        currentDetailDate = s.date;
        currentDetailTime = s.time;
        currentDetailSegments = (t && t.segments) ? t.segments : [];

        detailHeader.innerHTML =
            "<h2>" + formatDateTime(s.date, s.time) + "</h2>" +
            '<div class="meta">' +
            "<span>Duration: " + formatDuration(s.duration_seconds) + "</span>" +
            (s.language ? "<span>Language: " + escapeHtml(s.language) + "</span>" : "") +
            (s.model ? "<span>Model: " + escapeHtml(s.model) + "</span>" : "") +
            (s.audio_size_mb ? "<span>Size: " + s.audio_size_mb.toFixed(1) + " MB</span>" : "") +
            "</div>";

        // Tag info in detail
        var tags = (t && t.session && t.session.tags) ? t.session.tags : null;
        if (tags) {
            var tagsHtml = '<div class="detail-tags">';
            if (tags.activity) tagsHtml += '<span class="tag-badge activity">' + escapeHtml(tags.activity.replace(/_/g, " ")) + '</span>';
            if (tags.intent) tagsHtml += '<span class="tag-badge activity">' + escapeHtml(tags.intent.replace(/_/g, " ")) + '</span>';
            if (tags.mood) tagsHtml += '<span class="tag-badge mood-' + tags.mood + '">' + escapeHtml(tags.mood) + '</span>';
            if (tags.quality) tagsHtml += '<span class="tag-badge topic">' + escapeHtml(tags.quality) + '</span>';
            if (tags.topics) {
                for (var tti = 0; tti < tags.topics.length; tti++) {
                    tagsHtml += '<span class="tag-badge topic">' + escapeHtml(tags.topics[tti]) + '</span>';
                }
            }
            if (tags.entities) {
                for (var tei = 0; tei < tags.entities.length; tei++) {
                    tagsHtml += '<span class="tag-badge activity">' + escapeHtml(tags.entities[tei]) + '</span>';
                }
            }
            tagsHtml += '</div>';
            detailHeader.innerHTML += tagsHtml;
            if (tags.summary) {
                detailHeader.innerHTML += '<div class="detail-summary">' + escapeHtml(tags.summary) + '</div>';
            }
        }

        // User tags section in detail modal
        currentSessionUserTags = (tags && tags.user_tags) ? tags.user_tags.slice() : [];
        renderDetailUserTags();

        var audioFile = (t && t.session && t.session.audio_file) ? t.session.audio_file : "audio.ogg";
        audioContainer.innerHTML =
            '<audio id="audio-el" controls preload="metadata" src="/audio/' + s.date + "/" + s.time + '/' + audioFile + '"></audio>';
        audioEl = document.getElementById("audio-el");

        audioEl.addEventListener("timeupdate", onTimeUpdate);

        // Always show voiceprint bar (users can create new speakers inline)
        var vpBar = document.getElementById("voiceprint-bar");
        var filterBar = document.getElementById("segment-filter-bar");
        var hasTagged = currentDetailSegments.some(function (seg) { return seg.person_id; });
        updateVoiceprintSpeakerSelect();
        // Reset new speaker form
        var newSp = document.getElementById("voiceprint-new-speaker");
        if (newSp) newSp.classList.add("hidden");
        if (hasTagged) {
            filterBar.classList.remove("hidden");
        } else {
            filterBar.classList.add("hidden");
        }

        renderSegments();
    }

    function renderSegments() {
        var segments = currentDetailSegments;
        if (segments.length === 0) {
            segmentsContainer.innerHTML = '<div class="loading">No segments</div>';
            return;
        }

        var html = "";
        segments.forEach(function (seg, idx) {
            var isUnknown = seg.person_id === "unknown";
            var isHidden = false;
            if (segmentFilter === "unknown-only" && !isUnknown && seg.person_id) isHidden = true;
            if (segmentFilter === "hide-unknown" && isUnknown) isHidden = true;

            var cls = "segment";
            if (isUnknown) cls += " unknown-segment";
            if (isHidden) cls += " hidden";

            html += '<div class="' + cls + '" data-idx="' + idx + '" data-start="' + seg.start + '" data-end="' + seg.end + '">';
            html += '<input type="checkbox" class="segment-checkbox" data-idx="' + idx + '">';
            html += '<span class="segment-time" data-seek="' + seg.start + '">' + formatTimestamp(seg.start) + "</span>";

            // Speaker badge
            if (seg.person_id && seg.person_id !== "unknown") {
                var badgeCls = seg.is_user ? "speaker-badge user-speaker" : "speaker-badge known-speaker";
                html += '<span class="' + badgeCls + '">' + escapeHtml(seg.person_name || seg.person_id) + "</span>";
            } else if (seg.person_id === "unknown") {
                var unknownLabel = seg.speaker ? ("? " + seg.speaker) : "?";
                html += '<span class="speaker-badge unknown-speaker">' + escapeHtml(unknownLabel) + '</span>';
            } else {
                html += '<span class="segment-speaker">' + (seg.speaker || "") + "</span>";
            }

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

        // Checkbox events
        var checkboxes = segmentsContainer.querySelectorAll(".segment-checkbox");
        for (var j = 0; j < checkboxes.length; j++) {
            checkboxes[j].addEventListener("change", updateVoiceprintSelection);
        }
    }

    function updateVoiceprintSelection() {
        var checked = segmentsContainer.querySelectorAll(".segment-checkbox:checked");
        var countEl = document.getElementById("voiceprint-selected-count");
        var addBtn = document.getElementById("voiceprint-add-btn");
        countEl.textContent = checked.length + " selected";
        addBtn.disabled = checked.length === 0;
    }

    function updateVoiceprintSpeakerSelect() {
        var sel = document.getElementById("voiceprint-speaker-select");
        sel.innerHTML = '<option value="">-- Speaker --</option>';
        speakersCache.forEach(function (sp) {
            var opt = document.createElement("option");
            opt.value = sp.id;
            opt.textContent = sp.name + (sp.is_user ? " (me)" : "");
            sel.appendChild(opt);
        });
        var newOpt = document.createElement("option");
        newOpt.value = "__new__";
        newOpt.textContent = "+ New Speaker...";
        sel.appendChild(newOpt);
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

    function loadPage(page, retryCount) {
        currentPage = page;
        listEl.innerHTML = '<div class="loading">Loading...</div>';
        fetchSessions(page, currentQuery).then(renderSessions).catch(function (err) {
            console.error("loadPage failed:", err);
            var retry = retryCount || 0;
            if (retry < 3) {
                setTimeout(function () { loadPage(page, retry + 1); }, 1000 * (retry + 1));
            } else {
                listEl.innerHTML = '<div class="empty-state">Failed to load. <a href="#" id="retry-link">Retry</a></div>';
                var retryLink = document.getElementById("retry-link");
                if (retryLink) retryLink.addEventListener("click", function (e) { e.preventDefault(); loadPage(1); });
            }
        });
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
        document.getElementById("detail-user-tags").innerHTML = "";
        segmentFilter = "all";
        modal.classList.remove("hidden");

        // Load speakers list, user tags, and session detail in parallel
        var speakersPromise = loadSpeakersCache();
        var userTagsPromise = loadUserTagsCache();
        var detailPromise = fetchSessionDetail(date, time);
        Promise.all([speakersPromise, userTagsPromise, detailPromise]).then(function (results) {
            renderDetail(results[2]);
        });
    }

    function loadSpeakersCache() {
        return fetch("/api/speakers/list").then(function (r) {
            if (!r.ok) return [];
            return r.json();
        }).then(function (data) {
            speakersCache = data || [];
        }).catch(function () {
            speakersCache = [];
        });
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
    if (filterActivity) filterActivity.addEventListener("change", function () { loadPage(1); });
    if (filterMood) filterMood.addEventListener("change", function () { loadPage(1); });
    if (filterUserTag) filterUserTag.addEventListener("change", function () { loadPage(1); });
    modalClose.addEventListener("click", closeModal);
    modal.addEventListener("click", function (e) {
        if (e.target === modal) closeModal();
    });

    // ============================================================
    // Settings panel
    // ============================================================

    function openSettings() {
        settingsModal.classList.remove("hidden");
        loadTerms();
        loadAutoDetected();
        loadSuggestions();
        loadSuggestionCount();

        loadHotwordsSettings();
        loadVoiceSettings();
        loadUserTagsSettings();
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

    // ============================================================
    // Batch Diarization
    // ============================================================

    // ============================================================
    // Unified Batch Progress Panel
    // ============================================================

    var diarizeBtn = document.getElementById("diarize-btn");
    var tagBtn = document.getElementById("tag-btn");
    var batchPanel = document.getElementById("batch-panel");
    var batchStatusText = document.getElementById("batch-status-text");
    var batchProgressText = document.getElementById("batch-progress-text");
    var batchCancelBtn = document.getElementById("batch-cancel-btn");
    var batchProgressFill = document.getElementById("batch-progress-fill");
    var batchCurrent = document.getElementById("batch-current");
    var batchTimer = null;
    var batchCancelTime = null;
    var batchFadeTimer = null;
    var activeBatchType = null; // "diarize" | "tags"

    var batchApiMap = {
        diarize: { start: "/api/diarize/start", status: "/api/diarize/status", cancel: "/api/diarize/cancel" },
        tags:    { start: "/api/tags/start",    status: "/api/tags/status",    cancel: "/api/tags/cancel"    },
    };

    var batchLabels = {
        diarize: { running: "話者分離中...", cancelling: "キャンセル中...", done: "話者分離完了" },
        tags:    {
            session_tag: "LLM タグ付け中...",
            voice_tag: "声紋タグ付け中...",
            cancelling: "キャンセル中...",
            done: "タグ付け完了",
        },
    };

    function showBatchPanel(type, state) {
        activeBatchType = type;
        batchPanel.classList.remove("hidden", "fade-out");
        if (batchFadeTimer) { clearTimeout(batchFadeTimer); batchFadeTimer = null; }
        updateBatchUI(type, state);
    }

    function hideBatchPanel() {
        if (batchFadeTimer) return; // already scheduled
        batchFadeTimer = setTimeout(function () {
            batchPanel.classList.add("fade-out");
            setTimeout(function () {
                batchPanel.classList.add("hidden");
                batchPanel.classList.remove("fade-out");
                batchFadeTimer = null;
                activeBatchType = null;
            }, 500);
        }, 5000);
    }

    function updateBatchUI(type, state) {
        var done = (state.processed || 0) + (state.failed || 0);
        var total = state.total || 0;
        var pct = total > 0 ? Math.round(done / total * 100) : 0;

        batchProgressFill.style.width = pct + "%";
        batchProgressText.textContent = done + " / " + total;

        if (state.running) {
            if (state.cancel_requested) {
                batchStatusText.textContent = (batchLabels[type] || {}).cancelling || "キャンセル中...";
                batchCancelBtn.style.display = "none";
                if (batchCancelTime && Date.now() - batchCancelTime > 30000) {
                    batchStatusText.textContent = "キャンセル待ち (現在のセッション完了待ち)";
                }
            } else {
                // Phase-aware label for tags
                if (type === "tags" && state.phase) {
                    var phaseLabel = batchLabels.tags[state.phase] || "処理中...";
                    // Show phase-specific progress
                    if (state.phase === "session_tag") {
                        var stDone = (state.session_tag_processed || 0) + (state.session_tag_failed || 0);
                        batchStatusText.textContent = phaseLabel;
                        batchProgressText.textContent = stDone + " / " + (state.session_tag_total || 0);
                        pct = state.session_tag_total > 0 ? Math.round(stDone / state.session_tag_total * 100) : 0;
                        batchProgressFill.style.width = pct + "%";
                    } else if (state.phase === "voice_tag") {
                        var vtDone = (state.voice_tag_processed || 0) + (state.voice_tag_failed || 0);
                        batchStatusText.textContent = phaseLabel;
                        batchProgressText.textContent = vtDone + " / " + (state.voice_tag_total || 0);
                        pct = state.voice_tag_total > 0 ? Math.round(vtDone / state.voice_tag_total * 100) : 0;
                        batchProgressFill.style.width = pct + "%";
                    }
                } else {
                    batchStatusText.textContent = (batchLabels[type] || {}).running || "処理中...";
                }
                batchCancelBtn.style.display = "";
            }
            batchCurrent.textContent = state.current_session || "";
        } else {
            batchCancelBtn.style.display = "none";
            batchCurrent.textContent = "";
            batchCancelTime = null;
            if (state.finished_at) {
                var label = (batchLabels[type] || {}).done || "完了";
                var msg = label + ": " + (state.processed || 0) + " 件処理";
                if (state.failed > 0) msg += ", " + state.failed + " 件失敗";
                if (state.cancel_requested) msg += " (キャンセル)";
                batchStatusText.textContent = msg;
            }
        }
    }

    function startBatchPolling(type) {
        if (batchTimer) return;
        var api = batchApiMap[type];
        if (!api) return;
        var errorCount = 0;
        batchTimer = setInterval(function () {
            fetch(api.status).then(function (r) { return r.json(); }).then(function (state) {
                errorCount = 0;
                updateBatchUI(type, state);
                if (!state.running) {
                    clearInterval(batchTimer);
                    batchTimer = null;
                    loadPage(currentPage);
                    hideBatchPanel();
                }
            }).catch(function () {
                errorCount++;
                // Only stop polling after 15 consecutive failures (30s)
                if (errorCount >= 15) {
                    clearInterval(batchTimer);
                    batchTimer = null;
                }
            });
        }, 2000);
    }

    function startBatch(type, force) {
        var api = batchApiMap[type];
        if (!api) return;
        var opts = { method: "POST" };
        if (force) {
            opts.headers = { "Content-Type": "application/json" };
            opts.body = JSON.stringify({ force: true });
        }
        fetch(api.start, opts).then(function (r) {
            return r.json().then(function (data) {
                if (r.ok) {
                    if (data.status === "no_sessions") {
                        if (confirm("すべてのセッションは処理済みです。再実行しますか？")) {
                            startBatch(type, true);
                        }
                        return;
                    }
                    showBatchPanel(type, { running: true, total: data.total || 0, processed: 0, failed: 0 });
                    startBatchPolling(type);
                } else if (r.status === 409) {
                    alert("既に実行中です");
                } else if (r.status === 503) {
                    alert("必要な設定がありません: " + (data.error || ""));
                } else {
                    alert("エラー: " + (data.error || r.status));
                }
            });
        }).catch(function (e) {
            alert("開始に失敗しました: " + e.message);
        });
    }

    function checkBatchStatus(type) {
        var api = batchApiMap[type];
        if (!api) return;
        fetch(api.status).then(function (r) {
            if (r.ok) {
                if (type === "diarize") diarizeBtn.classList.remove("hidden");
                return r.json();
            }
            if (type === "diarize") diarizeBtn.classList.add("hidden");
            return null;
        }).then(function (state) {
            if (state && state.running) {
                showBatchPanel(type, state);
                startBatchPolling(type);
            }
        }).catch(function () {
            if (type === "diarize") diarizeBtn.classList.add("hidden");
        });
    }

    diarizeBtn.addEventListener("click", function () {
        if (!confirm("話者分離を実行しますか？")) return;
        startBatch("diarize", false);
    });

    tagBtn.addEventListener("click", function () {
        if (!confirm("タグ付け (LLM + 声紋) を実行しますか？")) return;
        startBatch("tags", false);
    });

    batchCancelBtn.addEventListener("click", function () {
        if (!activeBatchType) return;
        var api = batchApiMap[activeBatchType];
        if (!api) return;
        batchCancelTime = Date.now();
        batchCancelBtn.style.display = "none";
        batchStatusText.textContent = "キャンセル中...";
        fetch(api.cancel, { method: "POST" }).catch(function () {});
    });

    // ============================================================
    // Hotwords
    // ============================================================

    var hotwordsEnabledCb = document.getElementById("hotwords-enabled");
    var hotwordsMaxTotal = document.getElementById("hotwords-max-total");
    var hotwordsCategoriesEl = document.getElementById("hotwords-categories");
    var hotwordsCustomInput = document.getElementById("hotword-custom-input");
    var hotwordsCustomAddBtn = document.getElementById("hotword-custom-add-btn");
    var hotwordsCustomListEl = document.getElementById("hotwords-custom-list");
    var hotwordsTotalCount = document.getElementById("hotwords-total-count");
    var saveHotwordsBtn = document.getElementById("save-hotwords-settings");
    var hotwordsStatus = document.getElementById("hotwords-status");
    var hotwordsBadge = document.getElementById("hotwords-badge");

    var hotwordsSettingsCache = null;
    var hotwordsStatusCache = null;

    function loadHotwordsStatus() {
        fetch("/api/hotwords/status").then(function (r) {
            if (!r.ok) return null;
            return r.json();
        }).then(function (data) {
            if (!data) return;
            hotwordsStatusCache = data;
            hotwordsBadge.textContent = data.count || 0;
            hotwordsTotalCount.textContent = (data.count || 0) + " active";
        }).catch(function () {});
    }

    function loadHotwordsSettings() {
        fetch("/api/hotwords/settings").then(function (r) {
            if (!r.ok) return null;
            return r.json();
        }).then(function (data) {
            if (!data) return;
            hotwordsSettingsCache = data;
            hotwordsEnabledCb.checked = data.enabled !== false;
            hotwordsMaxTotal.value = data.max_total || 500;
            renderHotwordsCategories(data.categories || {});
            loadHotwordsCustom();
            loadHotwordsStatus();
        }).catch(function () {});
    }

    function renderHotwordsCategories(categories) {
        var catNames = {
            place: "Places", company: "Companies", station: "Stations",
            surname: "Surnames", given: "Given Names", product: "Products"
        };
        var html = "";
        for (var cat in categories) {
            var c = categories[cat];
            var label = catNames[cat] || cat;
            html += '<div class="hotword-category-row">';
            html += '<label class="toggle-label">';
            html += '<input type="checkbox" class="hw-cat-enabled" data-cat="' + cat + '"' +
                (c.enabled !== false ? " checked" : "") + '>';
            html += '<span>' + escapeHtml(label) + '</span>';
            html += '</label>';
            html += '<input type="number" class="hw-cat-max" data-cat="' + cat + '" ' +
                'min="0" max="2000" value="' + (c.max || 100) + '" style="width:60px">';
            html += '</div>';
        }
        hotwordsCategoriesEl.innerHTML = html;
    }

    function loadHotwordsCustom() {
        fetch("/api/hotwords/status").then(function (r) {
            if (!r.ok) return null;
            return r.json();
        }).then(function (data) {
            if (!data) return;
            hotwordsCustomCache = data.custom || [];
            renderHotwordsCustom();
        }).catch(function () {});
    }

    var hotwordsCustomCache = [];

    function renderHotwordsCustom() {
        if (hotwordsCustomCache.length === 0) {
            hotwordsCustomListEl.innerHTML = '<div class="text-dim">No custom hotwords</div>';
            return;
        }
        var html = "";
        hotwordsCustomCache.forEach(function (word) {
            html += '<div class="hotword-custom-item">';
            html += '<span>' + escapeHtml(word) + '</span>';
            html += '<button class="btn-danger btn-sm hw-custom-del" data-word="' + escapeHtml(word) + '">Del</button>';
            html += '</div>';
        });
        hotwordsCustomListEl.innerHTML = html;

        var delBtns = hotwordsCustomListEl.querySelectorAll(".hw-custom-del");
        for (var i = 0; i < delBtns.length; i++) {
            delBtns[i].addEventListener("click", function () {
                removeHotwordCustom(this.getAttribute("data-word"));
            });
        }
    }

    hotwordsCustomAddBtn.addEventListener("click", addHotwordCustom);
    hotwordsCustomInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") addHotwordCustom();
    });

    function addHotwordCustom() {
        var word = hotwordsCustomInput.value.trim();
        if (!word) return;
        fetch("/api/hotwords/custom", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ word: word }),
        }).then(function (r) { return r.json(); }).then(function (data) {
            if (data.custom) {
                hotwordsCustomCache = data.custom;
                renderHotwordsCustom();
            }
            hotwordsCustomInput.value = "";
            loadHotwordsStatus();
        });
    }

    function removeHotwordCustom(word) {
        fetch("/api/hotwords/custom", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ word: word }),
        }).then(function (r) { return r.json(); }).then(function (data) {
            if (data.custom) {
                hotwordsCustomCache = data.custom;
                renderHotwordsCustom();
            }
            loadHotwordsStatus();
        });
    }

    saveHotwordsBtn.addEventListener("click", function () {
        var categories = {};
        var enabledCbs = hotwordsCategoriesEl.querySelectorAll(".hw-cat-enabled");
        var maxInputs = hotwordsCategoriesEl.querySelectorAll(".hw-cat-max");

        for (var i = 0; i < enabledCbs.length; i++) {
            var cat = enabledCbs[i].getAttribute("data-cat");
            categories[cat] = { enabled: enabledCbs[i].checked };
        }
        for (var j = 0; j < maxInputs.length; j++) {
            var cat2 = maxInputs[j].getAttribute("data-cat");
            if (categories[cat2]) {
                categories[cat2].max = parseInt(maxInputs[j].value, 10) || 100;
            }
        }

        var payload = {
            enabled: hotwordsEnabledCb.checked,
            max_total: parseInt(hotwordsMaxTotal.value, 10) || 500,
            categories: categories,
        };

        fetch("/api/hotwords/settings", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        }).then(function (r) { return r.json(); }).then(function (data) {
            hotwordsStatus.textContent = "Saved!";
            setTimeout(function () { hotwordsStatus.textContent = ""; }, 2000);
            loadHotwordsStatus();
        });
    });

    // ============================================================
    // Voice / Speaker Management
    // ============================================================

    var voiceNameInput = document.getElementById("voice-name-input");
    var voiceIsUserCb = document.getElementById("voice-is-user");
    var voiceAddBtn = document.getElementById("voice-add-btn");
    var voiceSpeakersListEl = document.getElementById("voice-speakers-list");
    var voiceStatusText = document.getElementById("voice-status-text");
    var voiceThresholdSlider = document.getElementById("voice-threshold");
    var voiceThresholdValue = document.getElementById("voice-threshold-value");
    var voiceThresholdSaveBtn = document.getElementById("voice-threshold-save");
    var voiceDeleteAllBtn = document.getElementById("voice-delete-all-btn");
    var voiceStatus = document.getElementById("voice-status");

    function loadVoiceSettings() {
        fetch("/api/speakers/status").then(function (r) {
            if (!r.ok) return null;
            return r.json();
        }).then(function (data) {
            if (!data) {
                voiceStatusText.textContent = "Unavailable (no HF token)";
                return;
            }
            voiceStatusText.textContent = data.speaker_count + " speaker(s), " + data.total_samples + " sample(s)";
            voiceThresholdSlider.value = data.threshold || 0.55;
            voiceThresholdValue.textContent = (data.threshold || 0.55).toFixed(2);
        }).catch(function () {
            voiceStatusText.textContent = "Unavailable";
        });

        loadVoiceSpeakers();
    }

    function loadVoiceSpeakers() {
        fetch("/api/speakers/list").then(function (r) {
            if (!r.ok) return [];
            return r.json();
        }).then(function (data) {
            speakersCache = data || [];
            renderVoiceSpeakers(data || []);
        }).catch(function () {
            voiceSpeakersListEl.innerHTML = '<div class="text-dim">Unavailable</div>';
        });
    }

    function renderVoiceSpeakers(speakers) {
        if (speakers.length === 0) {
            voiceSpeakersListEl.innerHTML = '<div class="loading">No speakers registered</div>';
            return;
        }

        var html = "";
        speakers.forEach(function (sp) {
            html += '<div class="voice-speaker-row" data-id="' + escapeHtml(sp.id) + '">';
            html += '<div class="voice-speaker-header">';
            html += '<span class="voice-speaker-name">' + escapeHtml(sp.name) + '</span>';
            if (sp.is_user) {
                html += ' <span class="badge-user">Me</span>';
            } else {
                html += ' <span class="badge-other">Other</span>';
            }
            html += '<span class="voice-speaker-meta">' +
                sp.sample_count + ' sample(s), ' + (sp.total_duration || 0).toFixed(1) + 's' +
                (sp.has_embedding ? '' : ' (no embedding)') +
                '</span>';
            html += '<button class="btn-edit voice-speaker-edit" data-id="' + escapeHtml(sp.id) + '">Edit</button>';
            html += '<button class="btn-danger voice-speaker-del" data-id="' + escapeHtml(sp.id) + '">Del</button>';
            html += '</div>';
            html += '<div class="voice-samples-list" id="samples-' + escapeHtml(sp.id) + '">';
            html += '<a href="#" class="text-dim voice-show-samples" data-id="' + escapeHtml(sp.id) + '">Show samples</a>';
            html += '</div>';
            html += '</div>';
        });
        voiceSpeakersListEl.innerHTML = html;

        // Bind events
        var delBtns = voiceSpeakersListEl.querySelectorAll(".voice-speaker-del");
        for (var i = 0; i < delBtns.length; i++) {
            delBtns[i].addEventListener("click", function () {
                var id = this.getAttribute("data-id");
                if (!confirm("Delete this speaker and all voiceprints?")) return;
                fetch("/api/speakers/" + id, { method: "DELETE" })
                    .then(function (r) { return r.json(); })
                    .then(function () { loadVoiceSpeakers(); loadVoiceSettings(); });
            });
        }

        var editBtns = voiceSpeakersListEl.querySelectorAll(".voice-speaker-edit");
        for (var j = 0; j < editBtns.length; j++) {
            editBtns[j].addEventListener("click", function () {
                editSpeaker(this.getAttribute("data-id"));
            });
        }

        var showLinks = voiceSpeakersListEl.querySelectorAll(".voice-show-samples");
        for (var k = 0; k < showLinks.length; k++) {
            showLinks[k].addEventListener("click", function (e) {
                e.preventDefault();
                loadSpeakerSamples(this.getAttribute("data-id"));
            });
        }
    }

    function editSpeaker(id) {
        var sp = speakersCache.find(function (s) { return s.id === id; });
        if (!sp) return;
        var newName = prompt("Speaker name:", sp.name);
        if (!newName || !newName.trim()) return;
        fetch("/api/speakers/" + id, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: newName.trim() }),
        }).then(function (r) { return r.json(); })
          .then(function () { loadVoiceSpeakers(); });
    }

    function loadSpeakerSamples(speakerId) {
        // We need full speaker data — for now re-fetch from the status
        // The samples are stored in config/speakers.json, accessible via speakers/list
        // But list only returns summary. We need to read the samples from the file.
        // Actually, let's fetch the speakers.json config directly through a custom endpoint
        // or just show what we know from the summary.
        var el = document.getElementById("samples-" + speakerId);
        if (!el) return;
        el.innerHTML = '<div class="text-dim">Sample details in config/speakers.json</div>';
    }

    voiceAddBtn.addEventListener("click", function () {
        var name = voiceNameInput.value.trim();
        if (!name) return;
        var isUser = voiceIsUserCb.checked;
        fetch("/api/speakers/add", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: name, is_user: isUser }),
        }).then(function (r) { return r.json(); }).then(function () {
            voiceNameInput.value = "";
            voiceIsUserCb.checked = false;
            loadVoiceSpeakers();
            loadVoiceSettings();
        });
    });

    voiceNameInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") voiceAddBtn.click();
    });

    voiceThresholdSlider.addEventListener("input", function () {
        voiceThresholdValue.textContent = parseFloat(this.value).toFixed(2);
    });

    voiceThresholdSaveBtn.addEventListener("click", function () {
        var val = parseFloat(voiceThresholdSlider.value);
        fetch("/api/speakers/threshold", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ threshold: val }),
        }).then(function (r) { return r.json(); }).then(function () {
            voiceStatus.textContent = "Threshold saved!";
            setTimeout(function () { voiceStatus.textContent = ""; }, 2000);
        });
    });

    voiceDeleteAllBtn.addEventListener("click", function () {
        if (!confirm("Delete ALL speakers and voiceprints?")) return;
        fetch("/api/speakers/all", { method: "DELETE" })
            .then(function (r) { return r.json(); })
            .then(function () { loadVoiceSpeakers(); loadVoiceSettings(); });
    });

    // (Batch Speaker Tagging - removed, integrated into unified batch panel)

    // ============================================================
    // Voiceprint add from session detail
    // ============================================================

    var voiceprintAddBtn = document.getElementById("voiceprint-add-btn");
    var voiceprintSpeakerSelect = document.getElementById("voiceprint-speaker-select");
    var voiceprintNewSpeaker = document.getElementById("voiceprint-new-speaker");
    var voiceprintNewName = document.getElementById("voiceprint-new-name");
    var voiceprintNewIsUser = document.getElementById("voiceprint-new-is-user");

    // Show/hide new speaker fields when dropdown changes
    voiceprintSpeakerSelect.addEventListener("change", function () {
        if (this.value === "__new__") {
            voiceprintNewSpeaker.classList.remove("hidden");
            voiceprintNewName.focus();
        } else {
            voiceprintNewSpeaker.classList.add("hidden");
        }
    });

    function addVoiceprintToSpeaker(speakerId, indices, session) {
        voiceprintAddBtn.disabled = true;
        voiceprintAddBtn.textContent = "Adding...";

        fetch("/api/speakers/" + speakerId + "/voiceprint", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session: session, segment_indices: indices }),
        }).then(function (r) {
            return r.json().then(function (data) {
                if (r.ok) {
                    alert("Voiceprint added! (" + (data.duration || 0).toFixed(1) + "s)");
                    var cbs = segmentsContainer.querySelectorAll(".segment-checkbox:checked");
                    for (var j = 0; j < cbs.length; j++) cbs[j].checked = false;
                    updateVoiceprintSelection();
                    // Refresh speakers cache
                    loadSpeakersCache().then(function () { updateVoiceprintSpeakerSelect(); });
                } else {
                    alert("Error: " + (data.error || "unknown"));
                }
            });
        }).catch(function (e) {
            alert("Failed: " + e.message);
        }).finally(function () {
            voiceprintAddBtn.disabled = false;
            voiceprintAddBtn.textContent = "Add to voiceprint";
            voiceprintNewSpeaker.classList.add("hidden");
            voiceprintNewName.value = "";
            voiceprintNewIsUser.checked = false;
        });
    }

    voiceprintAddBtn.addEventListener("click", function () {
        var speakerId = voiceprintSpeakerSelect.value;
        if (!speakerId) {
            alert("Select a speaker or create a new one");
            return;
        }

        var checked = segmentsContainer.querySelectorAll(".segment-checkbox:checked");
        if (checked.length === 0) {
            alert("Select segments first");
            return;
        }

        var indices = [];
        for (var i = 0; i < checked.length; i++) {
            indices.push(parseInt(checked[i].getAttribute("data-idx"), 10));
        }
        var session = currentDetailDate + "/" + currentDetailTime;

        if (speakerId === "__new__") {
            // Create new speaker first, then add voiceprint
            var name = voiceprintNewName.value.trim();
            if (!name) {
                alert("Enter a speaker name");
                voiceprintNewName.focus();
                return;
            }
            var isUser = voiceprintNewIsUser.checked;
            voiceprintAddBtn.disabled = true;
            voiceprintAddBtn.textContent = "Creating...";

            fetch("/api/speakers/add", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: name, is_user: isUser }),
            }).then(function (r) { return r.json(); }).then(function (sp) {
                if (sp && sp.id) {
                    addVoiceprintToSpeaker(sp.id, indices, session);
                } else {
                    alert("Failed to create speaker");
                    voiceprintAddBtn.disabled = false;
                    voiceprintAddBtn.textContent = "Add to voiceprint";
                }
            }).catch(function (e) {
                alert("Failed to create speaker: " + e.message);
                voiceprintAddBtn.disabled = false;
                voiceprintAddBtn.textContent = "Add to voiceprint";
            });
        } else {
            addVoiceprintToSpeaker(speakerId, indices, session);
        }
    });

    // ============================================================
    // Segment filter
    // ============================================================

    var filterBtns = document.querySelectorAll(".seg-filter-btn");
    for (var fi = 0; fi < filterBtns.length; fi++) {
        filterBtns[fi].addEventListener("click", function () {
            filterBtns = document.querySelectorAll(".seg-filter-btn");
            for (var fj = 0; fj < filterBtns.length; fj++) filterBtns[fj].classList.remove("active");
            this.classList.add("active");
            segmentFilter = this.getAttribute("data-filter");
            renderSegments();
        });
    }

    // (Batch Session Tagging - removed, integrated into unified batch panel)

    // ============================================================
    // User Tags Management
    // ============================================================

    var userTagNameInput = document.getElementById("user-tag-name");
    var userTagColorInput = document.getElementById("user-tag-color");
    var userTagDescInput = document.getElementById("user-tag-desc");
    var userTagAddBtn = document.getElementById("user-tag-add-btn");
    var userTagsListEl = document.getElementById("user-tags-list");

    function loadUserTagsCache() {
        return fetch("/api/user-tags").then(function (r) { return r.json(); }).then(function (data) {
            userTagsCache = data || [];
            populateUserTagFilter();
        }).catch(function () {
            userTagsCache = [];
        });
    }

    function populateUserTagFilter() {
        if (!filterUserTag) return;
        var currentVal = filterUserTag.value;
        filterUserTag.innerHTML = '<option value="">User Tag</option>';
        userTagsCache.forEach(function (t) {
            var opt = document.createElement("option");
            opt.value = t.id;
            opt.textContent = t.name;
            filterUserTag.appendChild(opt);
        });
        filterUserTag.value = currentVal;
    }

    function loadUserTagsSettings() {
        fetch("/api/user-tags").then(function (r) { return r.json(); }).then(function (data) {
            userTagsCache = data || [];
            renderUserTagsSettings(data || []);
            populateUserTagFilter();
        }).catch(function () {
            userTagsListEl.innerHTML = '<div class="loading">Failed to load tags</div>';
        });
    }

    function renderUserTagsSettings(tags) {
        if (tags.length === 0) {
            userTagsListEl.innerHTML = '<div class="loading">No custom tags defined</div>';
            return;
        }
        var html = "";
        tags.forEach(function (t) {
            html += '<div class="user-tag-row" data-id="' + escapeHtml(t.id) + '">';
            html += '<span class="user-tag-badge" style="background:' + escapeHtml(t.color) + '">' + escapeHtml(t.name) + '</span>';
            html += '<span class="user-tag-desc">' + escapeHtml(t.description || "") + '</span>';
            html += '<button class="btn-edit user-tag-edit" data-id="' + escapeHtml(t.id) + '">Edit</button>';
            html += '<button class="btn-danger user-tag-del" data-id="' + escapeHtml(t.id) + '">Del</button>';
            html += '</div>';
        });
        userTagsListEl.innerHTML = html;

        var delBtns = userTagsListEl.querySelectorAll(".user-tag-del");
        for (var i = 0; i < delBtns.length; i++) {
            delBtns[i].addEventListener("click", function () {
                var id = this.getAttribute("data-id");
                if (!confirm("Delete this tag?")) return;
                fetch("/api/user-tags/" + id, { method: "DELETE" })
                    .then(function (r) { return r.json(); })
                    .then(function () { loadUserTagsSettings(); });
            });
        }

        var editBtns = userTagsListEl.querySelectorAll(".user-tag-edit");
        for (var j = 0; j < editBtns.length; j++) {
            editBtns[j].addEventListener("click", function () {
                editUserTag(this.getAttribute("data-id"));
            });
        }
    }

    function editUserTag(id) {
        var t = userTagsCache.find(function (tag) { return tag.id === id; });
        if (!t) return;
        var newName = prompt("Tag name:", t.name);
        if (!newName || !newName.trim()) return;
        var newDesc = prompt("Description:", t.description || "");
        fetch("/api/user-tags/" + id, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: newName.trim(), description: (newDesc || "").trim() }),
        }).then(function (r) { return r.json(); })
          .then(function () { loadUserTagsSettings(); });
    }

    if (userTagAddBtn) {
        userTagAddBtn.addEventListener("click", addUserTag);
    }
    if (userTagNameInput) {
        userTagNameInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") addUserTag();
        });
    }

    function addUserTag() {
        var name = userTagNameInput.value.trim();
        if (!name) return;
        var color = userTagColorInput.value || "#6c9bff";
        var desc = userTagDescInput.value.trim();
        fetch("/api/user-tags", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: name, color: color, description: desc }),
        }).then(function (r) { return r.json(); }).then(function () {
            userTagNameInput.value = "";
            userTagDescInput.value = "";
            loadUserTagsSettings();
        });
    }

    // --- Detail modal user tag chips ---

    function renderDetailUserTags() {
        var container = document.getElementById("detail-user-tags");
        if (!container) return;
        if (userTagsCache.length === 0) {
            container.innerHTML = "";
            return;
        }
        var html = '<div class="detail-user-tags">';
        userTagsCache.forEach(function (t) {
            var isActive = currentSessionUserTags.indexOf(t.id) !== -1;
            html += '<span class="user-tag-chip' + (isActive ? ' active' : '') + '" ' +
                'data-tag-id="' + escapeHtml(t.id) + '" ' +
                'style="background:' + escapeHtml(t.color) + '">' +
                escapeHtml(t.name) +
                '</span>';
        });
        html += '</div>';
        container.innerHTML = html;

        var chips = container.querySelectorAll(".user-tag-chip");
        for (var i = 0; i < chips.length; i++) {
            chips[i].addEventListener("click", function () {
                var tagId = this.getAttribute("data-tag-id");
                toggleDetailUserTag(tagId);
            });
        }
    }

    var _userTagSaveTimer = null;

    function toggleDetailUserTag(tagId) {
        var idx = currentSessionUserTags.indexOf(tagId);
        if (idx === -1) {
            currentSessionUserTags.push(tagId);
        } else {
            currentSessionUserTags.splice(idx, 1);
        }
        renderDetailUserTags();

        // Debounced save
        if (_userTagSaveTimer) clearTimeout(_userTagSaveTimer);
        _userTagSaveTimer = setTimeout(function () {
            saveSessionUserTags();
        }, 500);
    }

    function saveSessionUserTags() {
        if (!currentDetailDate || !currentDetailTime) return;
        fetch("/api/sessions/" + currentDetailDate + "/" + currentDetailTime + "/user-tags", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_tags: currentSessionUserTags }),
        }).then(function (r) { return r.json(); }).catch(function (e) {
            console.error("Failed to save user tags:", e);
        });
    }

    // --- Init ---
    loadUserTagsCache().then(function () {
        loadPage(1);
    });
    checkBatchStatus("diarize");
    checkBatchStatus("tags");
    loadHotwordsStatus();
})();
