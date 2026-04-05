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
    var segmentSpeakerFilter = "";

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

    // V2 filter elements (3-axis taxonomy)
    var filterConversationType = document.getElementById("filter-conversation-type");
    var filterOutcome = document.getElementById("filter-outcome");
    var filterDomain = document.getElementById("filter-domain");
    var filterImportanceV2 = document.getElementById("filter-importance-v2");
    var filterSpeaker = document.getElementById("filter-speaker");
    var filterAudioScene = document.getElementById("filter-audio-scene");

    // Taxonomy cache
    var taxonomyCache = null;

    function fetchSessions(page, query) {
        var url = "/api/sessions?page=" + page + "&per_page=20";
        if (query) url += "&q=" + encodeURIComponent(query);
        if (filterConversationType && filterConversationType.value) url += "&conversation_type=" + encodeURIComponent(filterConversationType.value);
        if (filterOutcome && filterOutcome.value) url += "&outcome=" + encodeURIComponent(filterOutcome.value);
        if (filterDomain && filterDomain.value) url += "&domain=" + encodeURIComponent(filterDomain.value);
        if (filterImportanceV2 && filterImportanceV2.value) url += "&importance_v2=" + encodeURIComponent(filterImportanceV2.value);
        if (filterSpeaker && filterSpeaker.value) url += "&speaker=" + encodeURIComponent(filterSpeaker.value);
        if (filterAudioScene && filterAudioScene.value) url += "&audio_scene=" + encodeURIComponent(filterAudioScene.value);
        return fetch(url).then(function (r) { return r.json(); });
    }

    function loadTaxonomy() {
        return fetch("/api/tag-taxonomy").then(function (r) { return r.json(); }).then(function (data) {
            taxonomyCache = data;
            populateFilterDropdowns(data);
        }).catch(function () {
            taxonomyCache = null;
        });
    }

    function populateFilterDropdowns(taxonomy) {
        if (!taxonomy) return;
        function fillSelect(sel, items, label) {
            if (!sel) return;
            var currentVal = sel.value;
            sel.innerHTML = '<option value="">' + label + '</option>';
            (items || []).forEach(function (item) {
                var opt = document.createElement("option");
                opt.value = item;
                opt.textContent = item;
                sel.appendChild(opt);
            });
            sel.value = currentVal;
        }
        fillSelect(filterConversationType, taxonomy.conversation_type, "種類");
        fillSelect(filterOutcome, taxonomy.outcomes, "成果物");
        fillSelect(filterDomain, taxonomy.domains, "事業領域");
        fillSelect(filterImportanceV2, taxonomy.importance, "重要度");
        // Export filter dropdowns
        fillSelect(document.getElementById("export-conversation-type"), taxonomy.conversation_type, "All");
        fillSelect(document.getElementById("export-domain"), taxonomy.domains, "All");
        fillSelect(document.getElementById("export-importance"), taxonomy.importance, "All");
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
            // Tag badges — V2 (3-axis) preferred, V1 fallback
            var hasV2 = !!s.tags_conversation_type;
            if (hasV2) {
                html += '<div class="session-tags">';
                html += '<span class="tag-badge conv-type">' + escapeHtml(s.tags_conversation_type) + '</span>';
                if (s.tags_outcomes) {
                    var outcomes = s.tags_outcomes.split(",");
                    for (var oi = 0; oi < outcomes.length; oi++) {
                        html += '<span class="tag-badge outcome">' + escapeHtml(outcomes[oi].trim()) + '</span>';
                    }
                }
                if (s.tags_domains) {
                    var domains = s.tags_domains.split(",");
                    for (var di = 0; di < domains.length; di++) {
                        html += '<span class="tag-badge domain">' + escapeHtml(domains[di].trim()) + '</span>';
                    }
                }
                if (s.tags_importance_v2) {
                    var impClass = s.tags_importance_v2 === "高" ? "imp-high" : s.tags_importance_v2 === "中" ? "imp-med" : "imp-low";
                    html += '<span class="tag-badge ' + impClass + '">' + escapeHtml(s.tags_importance_v2) + '</span>';
                }
                if (s.tags_external_audio != null && s.tags_external_audio >= 0.8) {
                    html += '<span class="tag-badge ext-audio" title="外部音声比率: ' + Math.round(s.tags_external_audio * 100) + '%">TV/外部音声</span>';
                }
                if (s.tags_audio_scene) {
                    var sceneTitle = 'PANNs CNN14: ' + s.tags_audio_scene;
                    if (s.tags_audio_scene_spatial) sceneTitle += ' / ' + s.tags_audio_scene_spatial;
                    if (s.tags_audio_scene_confidence) sceneTitle += ' (' + Math.round(s.tags_audio_scene_confidence * 100) + '%)';
                    html += '<span class="tag-badge audio-scene" title="' + escapeHtml(sceneTitle) + '">' + escapeHtml(s.tags_audio_scene) + '</span>';
                }
                html += '</div>';
            } else if (s.tags_activity || s.tags_mood || s.tags_topics) {
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

        // Tag info in detail — V2 preferred, V1 fallback
        var tags = (t && t.session && t.session.tags) ? t.session.tags : null;
        if (tags) {
            var isV2Detail = !!tags.conversation_type;
            var tagsHtml = '<div class="detail-tags">';
            if (isV2Detail) {
                tagsHtml += '<span class="tag-badge conv-type">' + escapeHtml(tags.conversation_type) + '</span>';
                if (tags.outcomes) {
                    for (var doi = 0; doi < tags.outcomes.length; doi++) {
                        tagsHtml += '<span class="tag-badge outcome">' + escapeHtml(tags.outcomes[doi]) + '</span>';
                    }
                }
                if (tags.domains) {
                    for (var ddi = 0; ddi < tags.domains.length; ddi++) {
                        tagsHtml += '<span class="tag-badge domain">' + escapeHtml(tags.domains[ddi]) + '</span>';
                    }
                }
                if (tags.importance) {
                    var dImpClass = tags.importance === "高" ? "imp-high" : tags.importance === "中" ? "imp-med" : "imp-low";
                    tagsHtml += '<span class="tag-badge ' + dImpClass + '">' + escapeHtml(tags.importance) + '</span>';
                }
                if (tags.quality) tagsHtml += '<span class="tag-badge topic">' + escapeHtml(tags.quality) + '</span>';
            } else {
                if (tags.activity) tagsHtml += '<span class="tag-badge activity">' + escapeHtml(tags.activity.replace(/_/g, " ")) + '</span>';
                if (tags.intent) tagsHtml += '<span class="tag-badge activity">' + escapeHtml(tags.intent.replace(/_/g, " ")) + '</span>';
                if (tags.mood) tagsHtml += '<span class="tag-badge mood-' + tags.mood + '">' + escapeHtml(tags.mood) + '</span>';
                if (tags.quality) tagsHtml += '<span class="tag-badge topic">' + escapeHtml(tags.quality) + '</span>';
                if (tags.topics) {
                    for (var tti = 0; tti < tags.topics.length; tti++) {
                        tagsHtml += '<span class="tag-badge topic">' + escapeHtml(tags.topics[tti]) + '</span>';
                    }
                }
            }
            if (tags.entities) {
                for (var tei = 0; tei < tags.entities.length; tei++) {
                    tagsHtml += '<span class="tag-badge activity">' + escapeHtml(tags.entities[tei]) + '</span>';
                }
            }
            if (isV2Detail) {
                tagsHtml += ' <button class="btn-sm tag-edit-btn" id="tag-edit-trigger">&#9998; 編集</button>';
            }
            tagsHtml += '</div>';
            detailHeader.innerHTML += tagsHtml;
            if (isV2Detail) {
                detailHeader.innerHTML += '<div id="tag-editor-container" style="display:none"></div>';
            }
            if (tags.summary) {
                detailHeader.innerHTML += '<div class="detail-summary">' + escapeHtml(tags.summary) + '</div>';
            }
            if (isV2Detail && tags.decision_memo) {
                detailHeader.innerHTML += '<div class="detail-decision-memo">&#9997; ' + escapeHtml(tags.decision_memo) + '</div>';
            }
            if (isV2Detail && tags.external_audio_ratio != null && tags.external_audio_ratio >= 0.5) {
                detailHeader.innerHTML += '<div class="detail-external-audio">&#128250; 外部音声比率: ' + Math.round(tags.external_audio_ratio * 100) + '% (unknown speaker)</div>';
            }
            // Wire up tag edit button
            var editBtn = document.getElementById("tag-edit-trigger");
            if (editBtn) {
                editBtn.addEventListener("click", function () {
                    showTagEditor(s.date, s.time, tags);
                });
            }
        }

        // Audio scene section
        if (s.tags_audio_scene) {
            var sceneHtml = '<div class="detail-audio-scene"><div class="scene-row">';
            sceneHtml += '<span class="scene-label">シーン:</span> ' + escapeHtml(s.tags_audio_scene);
            if (s.tags_audio_scene_spatial) {
                sceneHtml += ' <span class="scene-label">空間:</span> ' + escapeHtml(s.tags_audio_scene_spatial);
            }
            if (s.tags_audio_scene_confidence) {
                sceneHtml += ' <span style="color:#888">(' + Math.round(s.tags_audio_scene_confidence * 100) + '%)</span>';
            }
            sceneHtml += '</div></div>';
            detailHeader.innerHTML += sceneHtml;
        }

        var audioFile = (t && t.session && t.session.audio_file) ? t.session.audio_file : "audio.ogg";
        audioContainer.innerHTML =
            '<audio id="audio-el" controls preload="metadata" src="/audio/' + s.date + "/" + s.time + '/' + audioFile + '"></audio>';
        audioEl = document.getElementById("audio-el");

        audioEl.addEventListener("timeupdate", onTimeUpdate);

        // Reset segment speaker filter
        segmentSpeakerFilter = "";

        // Always show voiceprint bar (users can create new speakers inline)
        var vpBar = document.getElementById("voiceprint-bar");
        var filterBar = document.getElementById("segment-filter-bar");
        var hasTagged = currentDetailSegments.some(function (seg) { return seg.person_id; });

        // Populate segment speaker filter dropdown
        var segSpkSelect = document.getElementById("segment-speaker-filter");
        if (segSpkSelect) {
            segSpkSelect.innerHTML = '<option value="">-- Speaker --</option>';
            var spkNames = {};
            currentDetailSegments.forEach(function (seg) {
                if (seg.person_name && seg.person_name !== "unknown") {
                    spkNames[seg.person_name] = true;
                }
            });
            Object.keys(spkNames).sort().forEach(function (name) {
                var opt = document.createElement("option");
                opt.value = name;
                opt.textContent = name;
                segSpkSelect.appendChild(opt);
            });
            // Also add "unknown" option if any unknown segments exist
            if (currentDetailSegments.some(function (seg) { return seg.person_id === "unknown"; })) {
                var uOpt = document.createElement("option");
                uOpt.value = "__unknown__";
                uOpt.textContent = "unknown";
                segSpkSelect.appendChild(uOpt);
            }
        }
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

            // Speaker name filter
            if (segmentSpeakerFilter) {
                if (segmentSpeakerFilter === "__unknown__") {
                    if (seg.person_id !== "unknown") isHidden = true;
                } else {
                    if (seg.person_name !== segmentSpeakerFilter) isHidden = true;
                }
            }

            var cls = "segment";
            if (isUnknown) cls += " unknown-segment";
            if (isHidden) cls += " hidden";

            html += '<div class="' + cls + '" data-idx="' + idx + '" data-start="' + seg.start + '" data-end="' + seg.end + '">';
            html += '<input type="checkbox" class="segment-checkbox" data-idx="' + idx + '">';
            html += '<span class="segment-time" data-seek="' + seg.start + '">' + formatTimestamp(seg.start) + "</span>";

            // Speaker badge (clickable for correction)
            if (seg.person_id && seg.person_id !== "unknown") {
                var badgeCls = seg.is_user ? "speaker-badge user-speaker clickable" : "speaker-badge known-speaker clickable";
                html += '<span class="' + badgeCls + '" data-seg-idx="' + idx + '">' + escapeHtml(seg.person_name || seg.person_id) + "</span>";
            } else if (seg.person_id === "unknown") {
                var unknownLabel = seg.speaker ? ("? " + seg.speaker) : "?";
                html += '<span class="speaker-badge unknown-speaker clickable" data-seg-idx="' + idx + '">' + escapeHtml(unknownLabel) + '</span>';
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

        // Speaker badge click → correction dropdown
        var badges = segmentsContainer.querySelectorAll(".speaker-badge.clickable");
        for (var bi = 0; bi < badges.length; bi++) {
            badges[bi].addEventListener("click", function (e) {
                e.stopPropagation();
                var segIdx = parseInt(this.getAttribute("data-seg-idx"), 10);
                showCorrectionDropdown(this, segIdx);
            });
        }
    }

    function updateVoiceprintSelection() {
        var checked = segmentsContainer.querySelectorAll(".segment-checkbox:checked");
        var countEl = document.getElementById("voiceprint-selected-count");
        var addBtn = document.getElementById("voiceprint-add-btn");
        countEl.textContent = checked.length + " selected";
        addBtn.disabled = checked.length === 0;
    }

    // ============================================================
    // Speaker tag correction
    // ============================================================

    function showCorrectionDropdown(badgeEl, segIdx) {
        // Close any existing dropdown
        closeCorrectionDropdown();

        var dropdown = document.createElement("div");
        dropdown.className = "correction-dropdown";
        dropdown.id = "correction-dropdown";

        // "unknown" option
        var unknownOpt = document.createElement("div");
        unknownOpt.className = "correction-option";
        unknownOpt.textContent = "unknown (clear)";
        unknownOpt.addEventListener("click", function () {
            closeCorrectionDropdown();
            applySpeakerCorrection(segIdx, "unknown");
        });
        dropdown.appendChild(unknownOpt);

        // All registered speakers
        speakersCache.forEach(function (sp) {
            var opt = document.createElement("div");
            opt.className = "correction-option";
            opt.textContent = sp.name + (sp.is_user ? " (me)" : "");
            opt.addEventListener("click", function () {
                closeCorrectionDropdown();
                applySpeakerCorrection(segIdx, sp.id);
            });
            dropdown.appendChild(opt);
        });

        // Position relative to badge
        var rect = badgeEl.getBoundingClientRect();
        dropdown.style.top = (rect.bottom + window.scrollY + 2) + "px";
        dropdown.style.left = (rect.left + window.scrollX) + "px";
        document.body.appendChild(dropdown);

        // Close on outside click (delayed to avoid immediate close)
        setTimeout(function () {
            document.addEventListener("click", closeCorrectionDropdownOnClick);
        }, 0);
    }

    function closeCorrectionDropdown() {
        var existing = document.getElementById("correction-dropdown");
        if (existing) existing.remove();
        document.removeEventListener("click", closeCorrectionDropdownOnClick);
    }

    function closeCorrectionDropdownOnClick(e) {
        var dd = document.getElementById("correction-dropdown");
        if (dd && !dd.contains(e.target)) {
            closeCorrectionDropdown();
        }
    }

    function applySpeakerCorrection(segIdx, speakerId) {
        var session = currentDetailDate + "/" + currentDetailTime;

        // Visual feedback: dim the segment
        var segEl = segmentsContainer.querySelector('.segment[data-idx="' + segIdx + '"]');
        if (segEl) segEl.style.opacity = "0.5";

        fetch("/api/speakers/tag-correction", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session: session,
                segment_index: segIdx,
                speaker_id: speakerId,
            }),
        }).then(function (r) {
            return r.json().then(function (data) {
                if (r.ok) {
                    // Update in-memory segment
                    var seg = currentDetailSegments[segIdx];
                    seg.person_id = data.person_id;
                    seg.person_name = data.person_name;
                    seg.is_user = data.is_user;
                    seg.manually_tagged = true;
                    seg.speaker_similarity = 1.0;
                    renderSegments();

                    if (data.auto_added_sample) {
                        showNotification("Speaker corrected. Voiceprint sample auto-added (" +
                            (data.auto_added_sample.duration || 0).toFixed(1) + "s)");
                    } else {
                        showNotification("Speaker corrected.");
                    }
                } else {
                    alert("Error: " + (data.error || "unknown"));
                    if (segEl) segEl.style.opacity = "";
                }
            });
        }).catch(function (e) {
            alert("Failed: " + e.message);
            if (segEl) segEl.style.opacity = "";
        });
    }

    function showNotification(msg) {
        var el = document.createElement("div");
        el.className = "notification show";
        el.textContent = msg;
        document.body.appendChild(el);
        setTimeout(function () {
            el.classList.remove("show");
            setTimeout(function () { el.remove(); }, 400);
        }, 3000);
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
        segmentFilter = "all";
        segmentSpeakerFilter = "";
        modal.classList.remove("hidden");

        // Load speakers list and session detail in parallel
        var speakersPromise = loadSpeakersCache();
        var detailPromise = fetchSessionDetail(date, time);
        Promise.all([speakersPromise, detailPromise]).then(function (results) {
            renderDetail(results[1]);
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
    // Filter panel toggle
    var filterToggleBtn = document.getElementById("filter-toggle-btn");
    var filterPanel = document.getElementById("filter-panel");
    var filterClearBtn = document.getElementById("filter-clear-btn");
    var filterCountBadge = document.getElementById("filter-count-badge");
    var allFilterSelects = [filterConversationType, filterOutcome, filterDomain, filterImportanceV2, filterUserTag, filterSpeaker, filterAudioScene];

    function updateFilterCount() {
        var count = 0;
        allFilterSelects.forEach(function (sel) {
            if (sel && sel.value) {
                count++;
                sel.classList.add("active-filter");
            } else if (sel) {
                sel.classList.remove("active-filter");
            }
        });
        if (count > 0) {
            filterCountBadge.textContent = count;
            filterCountBadge.classList.remove("hidden");
            filterClearBtn.classList.remove("hidden");
        } else {
            filterCountBadge.classList.add("hidden");
            filterClearBtn.classList.add("hidden");
        }
    }

    if (filterToggleBtn) {
        filterToggleBtn.addEventListener("click", function () {
            var isHidden = filterPanel.classList.toggle("hidden");
            filterToggleBtn.classList.toggle("active", !isHidden);
        });
    }

    if (filterClearBtn) {
        filterClearBtn.addEventListener("click", function () {
            allFilterSelects.forEach(function (sel) {
                if (sel) sel.value = "";
            });
            updateFilterCount();
            loadPage(1);
        });
    }

    allFilterSelects.forEach(function (sel) {
        if (sel) sel.addEventListener("change", function () { updateFilterCount(); loadPage(1); });
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
        loadTerms();
        loadAutoDetected();
        loadSuggestions();
        loadSuggestionCount();

        loadHotwordsSettings();
        loadVoiceSettings();
        loadTaxonomySettings();
        loadRagSettings();
        loadRefinerSettings();
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
        diarize:   { start: "/api/diarize/start",   status: "/api/diarize/status",   cancel: "/api/diarize/cancel"   },
        tags:      { start: "/api/tags/start",       status: "/api/tags/status",       cancel: "/api/tags/cancel"     },
        sanitize:  { start: "/api/sanitize/start",   status: "/api/sanitize/status",   cancel: "/api/sanitize/cancel"  },
    };

    var batchLabels = {
        diarize: { running: "話者分離中...", cancelling: "キャンセル中...", done: "話者分離完了" },
        tags:    {
            voice_tag: "(1/2) 声紋タグ付け中...",
            session_tag: "(2/2) LLM タグ付け中...",
            cancelling: "キャンセル中...",
            done: "タグ付け完了",
        },
        sanitize: { running: "LLM 匿名化処理中...", cancelling: "キャンセル中...", done: "匿名化完了" },
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
                // Phase-aware label for tags and vibevoice
                if (type === "tags" && state.phase) {
                    var phaseLabel = (batchLabels[type] || {})[state.phase] || "処理中...";
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
                if (type === "sanitize") {
                    if (state.dropped > 0) msg += ", " + state.dropped + " 件除外(高リスク)";
                    msg += ", " + (state.records_generated || 0) + " レコード生成";
                }
                if (state.cancel_requested) msg += " (キャンセル)";
                batchStatusText.textContent = msg;
                // Auto-download sanitize output
                if (type === "sanitize" && state.output_file && state.records_generated > 0) {
                    window.location.href = "/api/sanitize/download?file=" + encodeURIComponent(state.output_file);
                }
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
        fetch("/api/speakers/list-detailed").then(function (r) {
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
            if (sp.samples && sp.samples.length > 0) {
                var maxVisible = 3;
                var hiddenCount = sp.samples.length > maxVisible ? sp.samples.length - maxVisible : 0;
                sp.samples.forEach(function (sample, idx) {
                    var addedDate = sample.added_at ? sample.added_at.split("T")[0] : "?";
                    var src = sample.source_session || "-";
                    var srcLabel = sample.source === "correction" ? "correction" : src;
                    var sessionKey = (sample.source_session || "").replace(" (correction)", "").trim();
                    var hasAudio = sessionKey && sessionKey.indexOf("/") !== -1;
                    var hidden = idx >= maxVisible;
                    html += '<div class="voice-sample-row"' + (hidden ? ' style="display:none" data-hidden="1"' : '') + '>';
                    if (hasAudio) {
                        var parts = sessionKey.split("/");
                        var audioUrl = "/audio/" + encodeURIComponent(parts[0]) + "/" + encodeURIComponent(parts[1]) + "/";
                        html += '<button class="btn-sm voice-sample-play" data-url="' + escapeHtml(audioUrl) +
                                 '" data-start="' + (sample.audio_start != null ? sample.audio_start : "") +
                                 '" data-end="' + (sample.audio_end != null ? sample.audio_end : "") +
                                 '" title="Play sample">&#9654;</button>';
                    }
                    html += '<span>' + escapeHtml(srcLabel) + '</span>';
                    html += '<span>' + (sample.duration || 0).toFixed(1) + 's</span>';
                    html += '<span>' + escapeHtml(addedDate) + '</span>';
                    html += '<button class="btn-danger voice-sample-del" data-speaker="' + escapeHtml(sp.id) +
                             '" data-sample="' + escapeHtml(sample.id) + '">Del</button>';
                    html += '</div>';
                });
                if (hiddenCount > 0) {
                    html += '<a href="#" class="voice-show-all" data-speaker="' + escapeHtml(sp.id) + '">+' + hiddenCount + ' more...</a>';
                }
            } else {
                html += '<div class="text-dim">No samples</div>';
            }
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

        // Bind sample play buttons
        var playBtns = voiceSpeakersListEl.querySelectorAll(".voice-sample-play");
        for (var p = 0; p < playBtns.length; p++) {
            playBtns[p].addEventListener("click", function () {
                playSampleInline(this);
            });
        }

        // Bind sample delete buttons
        var sampleDelBtns = voiceSpeakersListEl.querySelectorAll(".voice-sample-del");
        for (var sd = 0; sd < sampleDelBtns.length; sd++) {
            sampleDelBtns[sd].addEventListener("click", function () {
                var sid = this.getAttribute("data-speaker");
                var sampleId = this.getAttribute("data-sample");
                if (!confirm("Delete this voiceprint sample?")) return;
                fetch("/api/speakers/" + sid + "/voiceprint/" + sampleId, { method: "DELETE" })
                    .then(function (r) { return r.json(); })
                    .then(function () { loadVoiceSpeakers(); loadVoiceSettings(); });
            });
        }

        // Bind "show all" links
        var showAllLinks = voiceSpeakersListEl.querySelectorAll(".voice-show-all");
        for (var sa = 0; sa < showAllLinks.length; sa++) {
            showAllLinks[sa].addEventListener("click", function (e) {
                e.preventDefault();
                var spId = this.getAttribute("data-speaker");
                var container = document.getElementById("samples-" + spId);
                if (!container) return;
                var hidden = container.querySelectorAll('[data-hidden="1"]');
                for (var h = 0; h < hidden.length; h++) {
                    hidden[h].style.display = "";
                    hidden[h].removeAttribute("data-hidden");
                }
                this.remove();
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

    function playSampleInline(btn) {
        var url = btn.getAttribute("data-url");
        var startStr = btn.getAttribute("data-start");
        var endStr = btn.getAttribute("data-end");
        var row = btn.closest(".voice-sample-row");
        if (!row) return;

        // Check if this row already has a player (toggle off)
        var existingPlayer = row.nextElementSibling;
        if (existingPlayer && existingPlayer.classList.contains("sample-player")) {
            var audio = existingPlayer.querySelector("audio");
            if (audio) audio.pause();
            existingPlayer.remove();
            btn.textContent = "\u25B6";
            btn.classList.remove("playing");
            return;
        }

        // Remove any existing player elsewhere
        var oldPlayers = voiceSpeakersListEl.querySelectorAll(".sample-player");
        for (var i = 0; i < oldPlayers.length; i++) {
            var oldAudio = oldPlayers[i].querySelector("audio");
            if (oldAudio) oldAudio.pause();
            oldPlayers[i].remove();
        }
        var oldBtns = voiceSpeakersListEl.querySelectorAll(".voice-sample-play.playing");
        for (var j = 0; j < oldBtns.length; j++) {
            oldBtns[j].textContent = "\u25B6";
            oldBtns[j].classList.remove("playing");
        }

        // Create player div
        var playerDiv = document.createElement("div");
        playerDiv.className = "sample-player";
        var audioEl2 = document.createElement("audio");
        audioEl2.controls = true;
        audioEl2.preload = "metadata";
        audioEl2.src = url;
        playerDiv.appendChild(audioEl2);

        // Insert after the row
        row.parentNode.insertBefore(playerDiv, row.nextSibling);
        btn.textContent = "\u25A0";
        btn.classList.add("playing");

        var hasRange = startStr !== "" && endStr !== "";

        audioEl2.addEventListener("loadedmetadata", function () {
            if (hasRange) {
                audioEl2.currentTime = parseFloat(startStr);
            }
            audioEl2.play().catch(function () {});
        });

        if (hasRange) {
            var endTime = parseFloat(endStr);
            audioEl2.addEventListener("timeupdate", function () {
                if (audioEl2.currentTime >= endTime) {
                    audioEl2.pause();
                }
            });
        }

        audioEl2.addEventListener("ended", function () {
            btn.textContent = "\u25B6";
            btn.classList.remove("playing");
        });

        audioEl2.addEventListener("pause", function () {
            btn.textContent = "\u25B6";
            btn.classList.remove("playing");
        });

        audioEl2.addEventListener("play", function () {
            btn.textContent = "\u25A0";
            btn.classList.add("playing");
        });
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
            // Reset speaker filter when using button filter
            segmentSpeakerFilter = "";
            var segSpkSel = document.getElementById("segment-speaker-filter");
            if (segSpkSel) segSpkSel.value = "";
            renderSegments();
        });
    }

    // Segment speaker filter
    var segSpeakerFilter = document.getElementById("segment-speaker-filter");
    if (segSpeakerFilter) {
        segSpeakerFilter.addEventListener("change", function () {
            segmentSpeakerFilter = this.value;
            // Reset button filter to "all" when using speaker filter
            if (segmentSpeakerFilter) {
                segmentFilter = "all";
                filterBtns = document.querySelectorAll(".seg-filter-btn");
                for (var fk = 0; fk < filterBtns.length; fk++) {
                    filterBtns[fk].classList.remove("active");
                    if (filterBtns[fk].getAttribute("data-filter") === "all") {
                        filterBtns[fk].classList.add("active");
                    }
                }
            }
            renderSegments();
        });
    }

    // (Batch Session Tagging - removed, integrated into unified batch panel)

    // --- Audio scene labels for filter ---

    function loadAudioSceneLabels() {
        return fetch("/api/audio-scene-labels").then(function (r) { return r.json(); }).then(function (labels) {
            if (!filterAudioScene || !Array.isArray(labels)) return;
            var currentVal = filterAudioScene.value;
            while (filterAudioScene.options.length > 1) filterAudioScene.remove(1);
            labels.forEach(function (label) {
                var opt = document.createElement("option");
                opt.value = label;
                opt.textContent = label;
                filterAudioScene.appendChild(opt);
            });
            filterAudioScene.value = currentVal;
        }).catch(function (e) {
            console.error("Failed to load audio scene labels:", e);
        });
    }

    // --- Speaker names for session filter ---

    function loadSpeakerNames() {
        fetch("/api/speaker-names").then(function (r) { return r.json(); }).then(function (names) {
            if (!filterSpeaker || !Array.isArray(names)) return;
            var currentVal = filterSpeaker.value;
            filterSpeaker.innerHTML = '<option value="">Speaker</option>';
            names.forEach(function (name) {
                var opt = document.createElement("option");
                opt.value = name;
                opt.textContent = name;
                filterSpeaker.appendChild(opt);
            });
            filterSpeaker.value = currentVal;
        }).catch(function () {});
    }

    // ============================================================
    // Export tab
    // ============================================================

    var exportFormatEl = document.getElementById("export-format");
    var exportDateFrom = document.getElementById("export-date-from");
    var exportDateTo = document.getElementById("export-date-to");
    var exportSpeakerMode = document.getElementById("export-speaker-mode");
    var exportMinSegments = document.getElementById("export-min-segments");
    var exportConversationType = document.getElementById("export-conversation-type");
    var exportDomain = document.getElementById("export-domain");
    var exportImportance = document.getElementById("export-importance");
    var exportIncludeMonologue = document.getElementById("export-include-monologue");
    var exportStatsBtn = document.getElementById("export-stats-btn");
    var exportDownloadBtn = document.getElementById("export-download-btn");
    var exportPreview = document.getElementById("export-preview");
    var exportPreviewContent = document.getElementById("export-preview-content");
    var exportStatusEl = document.getElementById("export-status");

    // Update monologue checkbox default based on format
    if (exportFormatEl) {
        exportFormatEl.addEventListener("change", function () {
            var fmt = exportFormatEl.value;
            if (fmt === "openai" || fmt === "conversation") {
                exportIncludeMonologue.checked = false;
            } else {
                exportIncludeMonologue.checked = true;
            }
            // Update button label for safe_training
            if (exportDownloadBtn) {
                exportDownloadBtn.textContent = fmt === "safe_training" ? "Generate & Download" : "Download JSONL";
            }
        });
    }

    function getExportQueryString() {
        var params = "format=" + encodeURIComponent(exportFormatEl.value);
        if (exportDateFrom.value) params += "&date_from=" + encodeURIComponent(exportDateFrom.value);
        if (exportDateTo.value) params += "&date_to=" + encodeURIComponent(exportDateTo.value);

        var qualities = [];
        document.querySelectorAll(".export-quality:checked").forEach(function (cb) {
            qualities.push(cb.value);
        });
        if (qualities.length > 0) params += "&quality=" + encodeURIComponent(qualities.join(","));

        if (exportConversationType && exportConversationType.value) params += "&conversation_type=" + encodeURIComponent(exportConversationType.value);
        if (exportDomain && exportDomain.value) params += "&domain=" + encodeURIComponent(exportDomain.value);
        if (exportImportance && exportImportance.value) params += "&importance_v2=" + encodeURIComponent(exportImportance.value);

        params += "&speaker_mode=" + encodeURIComponent(exportSpeakerMode.value);
        params += "&min_segments=" + encodeURIComponent(exportMinSegments.value || "3");
        params += "&include_monologue=" + (exportIncludeMonologue.checked ? "true" : "false");
        return params;
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    if (exportStatsBtn) {
        exportStatsBtn.addEventListener("click", function () {
            exportStatsBtn.disabled = true;
            exportStatsBtn.textContent = "Loading...";
            fetch("/api/export/stats?" + getExportQueryString())
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.error) {
                        exportStatusEl.textContent = "Error: " + data.error;
                        exportStatusEl.style.color = "#e74c3c";
                        return;
                    }
                    exportPreview.classList.remove("hidden");
                    exportPreviewContent.innerHTML =
                        '<div class="export-preview-line"><strong>' + data.sessions + '</strong> sessions / <strong>' + data.segments + '</strong> segments</div>' +
                        '<div class="export-preview-line">Date: ' + escapeHtml(data.date_from || "N/A") + ' ~ ' + escapeHtml(data.date_to || "N/A") + '</div>' +
                        '<div class="export-preview-line">Est. size: <strong>' + formatBytes(data.est_size_bytes) + '</strong></div>';
                    exportDownloadBtn.disabled = data.sessions === 0;
                    exportStatusEl.textContent = "";
                })
                .catch(function (err) {
                    exportStatusEl.textContent = "Error: " + err.message;
                    exportStatusEl.style.color = "#e74c3c";
                })
                .finally(function () {
                    exportStatsBtn.disabled = false;
                    exportStatsBtn.textContent = "Preview Stats";
                });
        });
    }

    if (exportDownloadBtn) {
        exportDownloadBtn.addEventListener("click", function () {
            var fmt = exportFormatEl.value;
            var qs = getExportQueryString();
            if (fmt === "safe_training") {
                // Use batch sanitize flow via Bridge LLM
                exportDownloadBtn.disabled = true;
                fetch("/api/sanitize/start?" + qs, { method: "POST" })
                    .then(function (r) { return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; }); })
                    .then(function (res) {
                        if (res.ok) {
                            if (res.data.status === "no_sessions") {
                                alert("フィルタ条件に一致するセッションがありません");
                                return;
                            }
                            showBatchPanel("sanitize", { running: true, total: res.data.total || 0, processed: 0, failed: 0 });
                            startBatchPolling("sanitize");
                        } else if (res.status === 409) {
                            alert("既に実行中です");
                        } else if (res.status === 503) {
                            alert("GEMINI_API_KEY が設定されていません");
                        } else {
                            alert("エラー: " + (res.data.error || res.status));
                        }
                    })
                    .catch(function (e) { alert("開始に失敗しました: " + e.message); })
                    .finally(function () { exportDownloadBtn.disabled = false; });
            } else {
                window.location.href = "/api/export/training?" + qs;
            }
        });
    }

    // ============================================================
    // Taxonomy Settings tab
    // ============================================================

    var taxonomyEditCache = null;

    var taxonomyLoadingEl = document.getElementById("taxonomy-loading");
    var taxonomyErrorEl = document.getElementById("taxonomy-error");
    var taxonomyEditorEl = document.getElementById("taxonomy-editor");
    var taxonomyRetryLink = document.getElementById("taxonomy-retry-link");
    if (taxonomyRetryLink) {
        taxonomyRetryLink.addEventListener("click", function (e) {
            e.preventDefault();
            loadTaxonomySettings();
        });
    }

    function loadTaxonomySettings() {
        if (taxonomyLoadingEl) taxonomyLoadingEl.classList.remove("hidden");
        if (taxonomyErrorEl) taxonomyErrorEl.classList.add("hidden");
        if (taxonomyEditorEl) taxonomyEditorEl.classList.add("hidden");
        fetch("/api/tag-taxonomy").then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function (data) {
            taxonomyEditCache = JSON.parse(JSON.stringify(data));
            if (taxonomyLoadingEl) taxonomyLoadingEl.classList.add("hidden");
            if (taxonomyEditorEl) taxonomyEditorEl.classList.remove("hidden");
            renderTaxonomySettings();
            renderKeywordSuggestions();
        }).catch(function (err) {
            if (taxonomyLoadingEl) taxonomyLoadingEl.classList.add("hidden");
            if (taxonomyErrorEl) {
                taxonomyErrorEl.classList.remove("hidden");
                taxonomyErrorEl.querySelector("span") || (taxonomyErrorEl.innerHTML =
                    'タクソノミーの読み込みに失敗しました (' + escapeHtml(err.message) + ')。<a href="#" id="taxonomy-retry-link">再読み込み</a>');
            }
        });
    }

    function renderTaxonomySettings() {
        if (!taxonomyEditCache) return;
        var axes = [
            { key: "conversation_type", listId: "taxonomy-conv-type-list", countId: "taxonomy-conv-type-count" },
            { key: "outcomes", listId: "taxonomy-outcomes-list", countId: "taxonomy-outcomes-count" },
            { key: "domains", listId: "taxonomy-domains-list", countId: "taxonomy-domains-count" },
            { key: "importance", listId: "taxonomy-importance-list", countId: "taxonomy-importance-count" },
        ];
        axes.forEach(function (axis) {
            var listEl = document.getElementById(axis.listId);
            if (!listEl) return;
            var items = taxonomyEditCache[axis.key] || [];
            // Update count badge
            var countEl = document.getElementById(axis.countId);
            if (countEl) countEl.textContent = items.length;
            var html = "";
            items.forEach(function (item, idx) {
                html += '<div class="taxonomy-item">';
                html += '<span class="taxonomy-item-text">' + escapeHtml(item) + '</span>';
                html += '<button class="btn-danger btn-xs taxonomy-del" data-axis="' + axis.key + '" data-idx="' + idx + '">Del</button>';
                html += '</div>';
            });
            listEl.innerHTML = html || '<div class="text-dim">項目がありません</div>';

            var delBtns = listEl.querySelectorAll(".taxonomy-del");
            for (var i = 0; i < delBtns.length; i++) {
                delBtns[i].addEventListener("click", function () {
                    var ak = this.getAttribute("data-axis");
                    var ai = parseInt(this.getAttribute("data-idx"), 10);
                    taxonomyEditCache[ak].splice(ai, 1);
                    // If a domain was removed, also remove its keyword_anchors entry
                    if (ak === "domains" && taxonomyEditCache.keyword_anchors) {
                        var removedDomain = Object.keys(taxonomyEditCache.keyword_anchors).filter(function (d) {
                            return (taxonomyEditCache.domains || []).indexOf(d) === -1;
                        });
                        removedDomain.forEach(function (d) { delete taxonomyEditCache.keyword_anchors[d]; });
                    }
                    renderTaxonomySettings();
                });
            }
        });
        renderKeywordAnchors();
    }

    // --- Tag Editor (detail view) ---
    function showTagEditor(date, time, currentTags) {
        var container = document.getElementById("tag-editor-container");
        if (!container) return;
        if (container.style.display !== "none") {
            container.style.display = "none";
            return;
        }
        if (!taxonomyCache) { container.innerHTML = '<p class="text-dim">Taxonomy not loaded</p>'; container.style.display = "block"; return; }

        var html = '<div class="tag-editor">';
        // conversation_type select
        html += '<div class="tag-editor-row"><label>会話の種類</label><select id="te-conv-type">';
        (taxonomyCache.conversation_type || []).forEach(function (v) {
            html += '<option value="' + escapeHtml(v) + '"' + (currentTags.conversation_type === v ? ' selected' : '') + '>' + escapeHtml(v) + '</option>';
        });
        html += '</select></div>';
        // outcomes checkboxes
        html += '<div class="tag-editor-row"><label>成果物</label><div class="te-checks">';
        var curOutcomes = currentTags.outcomes || [];
        (taxonomyCache.outcomes || []).forEach(function (v) {
            var checked = curOutcomes.indexOf(v) !== -1 ? ' checked' : '';
            html += '<label class="te-check"><input type="checkbox" value="' + escapeHtml(v) + '"' + checked + ' class="te-outcome-cb"> ' + escapeHtml(v) + '</label>';
        });
        html += '</div></div>';
        // domains checkboxes
        html += '<div class="tag-editor-row"><label>事業領域</label><div class="te-checks">';
        var curDomains = currentTags.domains || [];
        (taxonomyCache.domains || []).forEach(function (v) {
            var checked = curDomains.indexOf(v) !== -1 ? ' checked' : '';
            html += '<label class="te-check"><input type="checkbox" value="' + escapeHtml(v) + '"' + checked + ' class="te-domain-cb"> ' + escapeHtml(v) + '</label>';
        });
        html += '</div></div>';
        // importance select
        html += '<div class="tag-editor-row"><label>重要度</label><select id="te-importance">';
        (taxonomyCache.importance || []).forEach(function (v) {
            html += '<option value="' + escapeHtml(v) + '"' + (currentTags.importance === v ? ' selected' : '') + '>' + escapeHtml(v) + '</option>';
        });
        html += '</select></div>';
        html += '<div class="tag-editor-actions"><button class="btn-primary" id="te-save">保存</button> <button class="btn-sm" id="te-cancel">キャンセル</button> <span id="te-status"></span></div>';
        html += '</div>';
        container.innerHTML = html;
        container.style.display = "block";

        document.getElementById("te-cancel").addEventListener("click", function () {
            container.style.display = "none";
        });
        document.getElementById("te-save").addEventListener("click", function () {
            var convType = document.getElementById("te-conv-type").value;
            var importance = document.getElementById("te-importance").value;
            var outcomes = [];
            var outCbs = container.querySelectorAll(".te-outcome-cb:checked");
            for (var i = 0; i < outCbs.length; i++) outcomes.push(outCbs[i].value);
            var domains = [];
            var domCbs = container.querySelectorAll(".te-domain-cb:checked");
            for (var j = 0; j < domCbs.length; j++) domains.push(domCbs[j].value);

            var statusEl = document.getElementById("te-status");
            statusEl.textContent = "Saving...";
            fetch("/api/sessions/" + encodeURIComponent(date) + "/" + encodeURIComponent(time) + "/session-tags", {
                method: "PUT",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({conversation_type: convType, outcomes: outcomes, domains: domains, importance: importance}),
            }).then(function (r) { return r.json(); }).then(function (data) {
                if (data.error) { statusEl.textContent = "Error: " + data.error; statusEl.style.color = "#e74c3c"; return; }
                statusEl.textContent = "Saved!";
                statusEl.style.color = "#2ecc71";
                // Refresh detail view
                setTimeout(function () {
                    container.style.display = "none";
                    loadPage(currentPage);
                    openDetail(date, time);
                }, 500);
            }).catch(function (err) {
                statusEl.textContent = "Error: " + err.message;
                statusEl.style.color = "#e74c3c";
            });
        });
    }

    // --- Keyword Suggestions (Settings > Taxonomy) ---
    function renderKeywordSuggestions() {
        var container = document.getElementById("keyword-suggestions-list");
        var countEl = document.getElementById("keyword-suggestions-count");
        if (!container) return;
        fetch("/api/keyword-suggestions").then(function (r) { return r.json(); }).then(function (data) {
            var suggestions = data.suggestions || [];
            if (countEl) countEl.textContent = suggestions.length;
            if (suggestions.length === 0) {
                container.innerHTML = '<div class="text-dim">候補なし</div>';
                return;
            }
            // Group by domain
            var groups = {};
            suggestions.forEach(function (s) {
                if (!groups[s.domain]) groups[s.domain] = [];
                groups[s.domain].push(s);
            });
            var html = '';
            Object.keys(groups).forEach(function (domain) {
                html += '<div class="kw-domain-group">';
                html += '<div class="kw-domain-name">' + escapeHtml(domain) + '</div>';
                html += '<div class="kw-suggestions-chips">';
                groups[domain].forEach(function (s) {
                    html += '<span class="kw-chip kw-suggestion" data-id="' + escapeHtml(s.id) + '">';
                    html += escapeHtml(s.keyword);
                    html += ' <span class="badge">' + s.occurrences + '</span>';
                    html += ' <span class="kw-approve-btn" data-id="' + escapeHtml(s.id) + '" title="承認">&#10003;</span>';
                    html += ' <span class="kw-reject-btn" data-id="' + escapeHtml(s.id) + '" title="拒否">&times;</span>';
                    html += '</span>';
                });
                html += '</div></div>';
            });
            container.innerHTML = html;

            // Approve handlers
            var approveBtns = container.querySelectorAll(".kw-approve-btn");
            for (var i = 0; i < approveBtns.length; i++) {
                approveBtns[i].addEventListener("click", function () {
                    var id = this.getAttribute("data-id");
                    fetch("/api/keyword-suggestions/" + encodeURIComponent(id) + "/approve", {method: "POST"})
                        .then(function (r) { return r.json(); })
                        .then(function () { renderKeywordSuggestions(); renderKeywordAnchors(); });
                });
            }
            // Reject handlers
            var rejectBtns = container.querySelectorAll(".kw-reject-btn");
            for (var j = 0; j < rejectBtns.length; j++) {
                rejectBtns[j].addEventListener("click", function () {
                    var id = this.getAttribute("data-id");
                    fetch("/api/keyword-suggestions/" + encodeURIComponent(id) + "/reject", {method: "POST"})
                        .then(function (r) { return r.json(); })
                        .then(function () { renderKeywordSuggestions(); });
                });
            }
        }).catch(function () {
            container.innerHTML = '<div class="text-dim">読み込みエラー</div>';
        });
    }

    function renderKeywordAnchors() {
        var container = document.getElementById("taxonomy-keywords-editor");
        if (!container || !taxonomyEditCache) return;
        var domains = taxonomyEditCache.domains || [];
        if (!taxonomyEditCache.keyword_anchors) taxonomyEditCache.keyword_anchors = {};
        var anchors = taxonomyEditCache.keyword_anchors;
        if (domains.length === 0) {
            container.innerHTML = '<div class="text-dim">事業領域が未登録です</div>';
            return;
        }
        var html = "";
        domains.forEach(function (domain) {
            var kws = anchors[domain] || [];
            html += '<div class="kw-domain-group">';
            html += '<div class="kw-domain-name">' + escapeHtml(domain) + '</div>';
            html += '<div class="kw-chips">';
            kws.forEach(function (kw, ki) {
                html += '<span class="kw-chip">' + escapeHtml(kw);
                html += ' <span class="kw-chip-del" data-domain="' + escapeHtml(domain) + '" data-idx="' + ki + '">&times;</span>';
                html += '</span>';
            });
            html += '</div>';
            html += '<div class="kw-add-row">';
            html += '<input type="text" class="kw-add-input" data-domain="' + escapeHtml(domain) + '" placeholder="キーワードを追加">';
            html += '<button class="btn-sm kw-add-btn" data-domain="' + escapeHtml(domain) + '">Add</button>';
            html += '</div>';
            html += '</div>';
        });
        container.innerHTML = html;

        // Delete handlers
        var delBtns = container.querySelectorAll(".kw-chip-del");
        for (var i = 0; i < delBtns.length; i++) {
            delBtns[i].addEventListener("click", function () {
                var d = this.getAttribute("data-domain");
                var idx = parseInt(this.getAttribute("data-idx"), 10);
                if (taxonomyEditCache.keyword_anchors[d]) {
                    taxonomyEditCache.keyword_anchors[d].splice(idx, 1);
                }
                renderKeywordAnchors();
            });
        }

        // Add handlers
        var addBtns = container.querySelectorAll(".kw-add-btn");
        for (var j = 0; j < addBtns.length; j++) {
            addBtns[j].addEventListener("click", function () {
                var d = this.getAttribute("data-domain");
                var input = container.querySelector('.kw-add-input[data-domain="' + d + '"]');
                if (!input) return;
                var val = input.value.trim();
                if (!val) return;
                if (!taxonomyEditCache.keyword_anchors[d]) taxonomyEditCache.keyword_anchors[d] = [];
                if (taxonomyEditCache.keyword_anchors[d].indexOf(val) === -1) {
                    taxonomyEditCache.keyword_anchors[d].push(val);
                }
                input.value = "";
                renderKeywordAnchors();
            });
        }

        // Enter key support for add inputs
        var addInputs = container.querySelectorAll(".kw-add-input");
        for (var k = 0; k < addInputs.length; k++) {
            addInputs[k].addEventListener("keydown", function (e) {
                if (e.key === "Enter") {
                    var d = this.getAttribute("data-domain");
                    var btn = container.querySelector('.kw-add-btn[data-domain="' + d + '"]');
                    if (btn) btn.click();
                }
            });
        }
    }

    // Add buttons
    var taxonomyAddBtns = document.querySelectorAll(".taxonomy-add-btn");
    for (var tai = 0; tai < taxonomyAddBtns.length; tai++) {
        taxonomyAddBtns[tai].addEventListener("click", function () {
            var axis = this.getAttribute("data-axis");
            var inputMap = {
                "conversation_type": "taxonomy-conv-type-input",
                "outcomes": "taxonomy-outcomes-input",
                "domains": "taxonomy-domains-input",
                "importance": "taxonomy-importance-input",
            };
            var input = document.getElementById(inputMap[axis]);
            if (!input) return;
            var val = input.value.trim();
            if (!val || !taxonomyEditCache) return;
            if (!taxonomyEditCache[axis]) taxonomyEditCache[axis] = [];
            if (taxonomyEditCache[axis].indexOf(val) === -1) {
                taxonomyEditCache[axis].push(val);
            }
            input.value = "";
            renderTaxonomySettings();
        });
    }

    // Save button
    var taxonomySaveBtn = document.getElementById("taxonomy-save-btn");
    var taxonomyStatusEl = document.getElementById("taxonomy-status");
    if (taxonomySaveBtn) {
        taxonomySaveBtn.addEventListener("click", function () {
            if (!taxonomyEditCache) return;
            taxonomySaveBtn.disabled = true;
            taxonomyStatusEl.textContent = "Saving...";
            taxonomyStatusEl.style.color = "";
            fetch("/api/tag-taxonomy", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ taxonomy: taxonomyEditCache }),
            }).then(function (r) { return r.json(); }).then(function (data) {
                if (data.error) {
                    taxonomyStatusEl.textContent = "Error: " + data.error;
                    taxonomyStatusEl.style.color = "#e74c3c";
                    return;
                }
                taxonomyStatusEl.textContent = "Saved!";
                taxonomyStatusEl.style.color = "#2ecc71";
                // Refresh filter dropdowns
                loadTaxonomy();
                // Check impact
                if (data.impact && Object.keys(data.impact).length > 0) {
                    var impactMsg = "以下のタグ値が使用中のセッションで削除されます:\n";
                    for (var k in data.impact) {
                        impactMsg += "  " + k + ": " + data.impact[k] + " セッション\n";
                    }
                    impactMsg += "\n再タグ付けを実行しますか？";
                    if (confirm(impactMsg)) {
                        fetch("/api/tag-taxonomy", {
                            method: "PUT",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ taxonomy: taxonomyEditCache, retag: true }),
                        }).then(function (r) { return r.json(); }).then(function (rd) {
                            if (rd.retag_started) {
                                taxonomyStatusEl.textContent = "Saved + retag started!";
                                showBatchPanel("tags", { running: true, total: 0, processed: 0 });
                                startBatchPolling("tags");
                            }
                        });
                    }
                }
                setTimeout(function () { taxonomyStatusEl.textContent = ""; }, 3000);
            }).catch(function (err) {
                taxonomyStatusEl.textContent = "Error: " + err.message;
                taxonomyStatusEl.style.color = "#e74c3c";
            }).finally(function () {
                taxonomySaveBtn.disabled = false;
            });
        });
    }

    // ============================================================
    // RAG Settings tab
    // ============================================================

    function loadRagSettings() {
        var loadingEl = document.getElementById("rag-loading");
        var infoEl = document.getElementById("rag-info");
        if (loadingEl) loadingEl.classList.remove("hidden");
        if (infoEl) infoEl.classList.add("hidden");

        fetch("/api/rag/status").then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function (data) {
            if (loadingEl) loadingEl.classList.add("hidden");
            if (infoEl) infoEl.classList.remove("hidden");

            var readyEl = document.getElementById("rag-ready-badge");
            if (readyEl) {
                if (!data.enabled) {
                    readyEl.innerHTML = '<span style="color:#95a5a6;">DISABLED</span>';
                } else if (data.ready) {
                    readyEl.innerHTML = '<span style="color:#2ecc71;">READY</span>';
                } else {
                    readyEl.innerHTML = '<span style="color:#f39c12;">NOT READY (' + (data.total_sessions || 0) + '/' + (data.min_sessions || 20) + ')</span>';
                }
            }
            var countEl = document.getElementById("rag-session-count");
            if (countEl) countEl.textContent = data.total_sessions || 0;
            var minEl = document.getElementById("rag-min-sessions");
            if (minEl) minEl.textContent = data.min_sessions || 20;
            var modelEl = document.getElementById("rag-model");
            if (modelEl) modelEl.textContent = data.model || "-";
            var kEl = document.getElementById("rag-k");
            if (kEl) kEl.textContent = data.k || "-";
            var confEl = document.getElementById("rag-confidence");
            if (confEl) confEl.textContent = data.confidence_threshold || "-";
        }).catch(function (err) {
            if (loadingEl) loadingEl.classList.add("hidden");
            if (infoEl) {
                infoEl.classList.remove("hidden");
                infoEl.innerHTML = '<div class="text-dim" style="color:#e74c3c;">RAG status unavailable: ' + escapeHtml(err.message) + '</div>';
            }
        });
    }

    var ragRebuildBtn = document.getElementById("rag-rebuild-btn");
    if (ragRebuildBtn) {
        ragRebuildBtn.addEventListener("click", function () {
            if (!confirm("RAG インデックスを再構築しますか？")) return;
            var statusMsg = document.getElementById("rag-status-msg");
            ragRebuildBtn.disabled = true;
            if (statusMsg) { statusMsg.textContent = "再構築中..."; statusMsg.style.color = "#f39c12"; }
            fetch("/api/rag/rebuild", { method: "POST" }).then(function (r) {
                return r.json();
            }).then(function (data) {
                if (data.error) {
                    if (statusMsg) { statusMsg.textContent = "Error: " + data.error; statusMsg.style.color = "#e74c3c"; }
                } else {
                    if (statusMsg) { statusMsg.textContent = "再構築開始しました"; statusMsg.style.color = "#2ecc71"; }
                    // Poll for completion
                    setTimeout(function () { loadRagSettings(); }, 5000);
                }
            }).catch(function (err) {
                if (statusMsg) { statusMsg.textContent = "Error: " + err.message; statusMsg.style.color = "#e74c3c"; }
            }).finally(function () {
                ragRebuildBtn.disabled = false;
                setTimeout(function () { if (statusMsg) statusMsg.textContent = ""; }, 5000);
            });
        });
    }

    // ============================================================
    // Refiner Settings
    // ============================================================

    function loadRefinerSettings() {
        var loadingEl = document.getElementById("refiner-loading");
        var infoEl = document.getElementById("refiner-info");
        var pendingPanel = document.getElementById("refiner-pending-panel");
        if (loadingEl) loadingEl.classList.remove("hidden");
        if (infoEl) infoEl.classList.add("hidden");
        if (pendingPanel) pendingPanel.classList.add("hidden");

        fetch("/api/taxonomy/refine/status").then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function (data) {
            if (loadingEl) loadingEl.classList.add("hidden");
            if (infoEl) infoEl.classList.remove("hidden");

            var badge = document.getElementById("refiner-enabled-badge");
            if (badge) {
                if (data.enabled) {
                    badge.textContent = "ENABLED";
                    badge.style.color = "#2ecc71";
                } else {
                    badge.textContent = "DISABLED";
                    badge.style.color = "#e74c3c";
                }
            }

            var runningEl = document.getElementById("refiner-running");
            if (runningEl) runningEl.textContent = data.running ? "Yes (" + (data.phase || "") + ")" : "No";

            var phaseEl = document.getElementById("refiner-phase");
            if (phaseEl) phaseEl.textContent = data.phase || "-";

            var lastRunEl = document.getElementById("refiner-last-run");
            if (lastRunEl) lastRunEl.textContent = data.last_run || "-";

            var pendingEl = document.getElementById("refiner-has-pending");
            if (pendingEl) pendingEl.textContent = data.has_pending ? "Yes" : "No";

            // Load pending if exists
            if (data.has_pending) {
                loadRefinerPending();
            }

            // Render reports
            var reportsList = document.getElementById("refiner-reports-list");
            if (reportsList && data.reports && data.reports.length > 0) {
                var html = "<table style='width:100%;font-size:0.85em;'><tr><th>Date</th><th>Status</th><th>Samples</th><th>Clusters</th></tr>";
                data.reports.forEach(function (r) {
                    html += "<tr><td>" + (r.filename || "-") + "</td><td>" + (r.status || "-") + "</td><td>" + (r.sample_count || 0) + "</td><td>" + (r.cluster_count || 0) + "</td></tr>";
                });
                html += "</table>";
                reportsList.innerHTML = html;
            }

            // If running, poll
            if (data.running) {
                setTimeout(loadRefinerSettings, 5000);
            }
        }).catch(function (err) {
            if (loadingEl) loadingEl.textContent = "Error: " + err.message;
        });
    }

    function loadRefinerPending() {
        var panel = document.getElementById("refiner-pending-panel");
        fetch("/api/taxonomy/refine/pending").then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function (data) {
            if (!data.pending) return;
            if (panel) panel.classList.remove("hidden");

            var changesList = document.getElementById("refiner-changes-list");
            if (changesList && data.pending.changes) {
                var html = "";
                data.pending.changes.forEach(function (c) {
                    var typeColor = {"split": "#3498db", "merge": "#e67e22", "new": "#2ecc71", "remove": "#e74c3c"}[c.type] || "#aaa";
                    html += "<div style='margin:4px 0;'><span style='color:" + typeColor + ";font-weight:bold;'>[" + c.type + "]</span> " + (c.axis || "") + ": " + (c.description || "") + "</div>";
                    if (c.before && c.before.length) html += "<div style='margin-left:16px;color:#999;'>Before: " + c.before.join(", ") + "</div>";
                    if (c.after && c.after.length) html += "<div style='margin-left:16px;color:#ccc;'>After: " + c.after.join(", ") + "</div>";
                });
                changesList.innerHTML = html || "<div class='text-dim'>変更なし</div>";
            }

            var validationEl = document.getElementById("refiner-validation");
            if (validationEl && data.pending.validation) {
                var v = data.pending.validation;
                validationEl.innerHTML = "<table style='font-size:0.85em;'>"
                    + "<tr><td>Accuracy</td><td>" + ((v.accuracy_vs_current * 100).toFixed(1)) + "%</td></tr>"
                    + "<tr><td>Improved</td><td style='color:#2ecc71;'>" + (v.improved_count || 0) + "</td></tr>"
                    + "<tr><td>Degraded</td><td style='color:#e74c3c;'>" + (v.degraded_count || 0) + "</td></tr>"
                    + "<tr><td>Unchanged</td><td>" + (v.unchanged_count || 0) + "</td></tr>"
                    + "</table>";
            }
        }).catch(function () {});
    }

    // Refiner buttons
    var refinerRunBtn = document.getElementById("refiner-run-btn");
    if (refinerRunBtn) {
        refinerRunBtn.addEventListener("click", function () {
            if (!confirm("タクソノミー精緻化パイプラインを実行しますか？（Batch API の完了まで数時間かかる場合があります）")) return;
            var statusMsg = document.getElementById("refiner-status-msg");
            refinerRunBtn.disabled = true;
            fetch("/api/taxonomy/refine/run", { method: "POST" }).then(function (r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            }).then(function () {
                if (statusMsg) { statusMsg.textContent = "Started"; statusMsg.style.color = "#2ecc71"; }
                setTimeout(loadRefinerSettings, 3000);
            }).catch(function (err) {
                if (statusMsg) { statusMsg.textContent = "Error: " + err.message; statusMsg.style.color = "#e74c3c"; }
            }).finally(function () {
                refinerRunBtn.disabled = false;
                setTimeout(function () { if (statusMsg) statusMsg.textContent = ""; }, 5000);
            });
        });
    }

    var refinerApproveBtn = document.getElementById("refiner-approve-btn");
    if (refinerApproveBtn) {
        refinerApproveBtn.addEventListener("click", function () {
            if (!confirm("提案を承認してタクソノミーを更新しますか？")) return;
            fetch("/api/taxonomy/refine/approve", { method: "POST" }).then(function (r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            }).then(function () {
                alert("承認しました。タクソノミーが更新されました。");
                loadRefinerSettings();
                loadTaxonomySettings();
            }).catch(function (err) {
                alert("Error: " + err.message);
            });
        });
    }

    var refinerRejectBtn = document.getElementById("refiner-reject-btn");
    if (refinerRejectBtn) {
        refinerRejectBtn.addEventListener("click", function () {
            if (!confirm("提案を却下しますか？")) return;
            fetch("/api/taxonomy/refine/reject", { method: "POST" }).then(function (r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            }).then(function () {
                alert("却下しました。");
                loadRefinerSettings();
            }).catch(function (err) {
                alert("Error: " + err.message);
            });
        });
    }

    // --- Init ---
    Promise.all([loadSpeakerNames(), loadTaxonomy(), loadAudioSceneLabels()]).then(function () {
        loadPage(1);
    });
    checkBatchStatus("diarize");
    checkBatchStatus("tags");
    loadHotwordsStatus();
})();
