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

        // Audio scene section (with extended dimensions)
        if (s.tags_audio_scene || s.tags_audio_noise_sources || s.tags_audio_events || s.tags_audio_speaker_count) {
            var sceneHtml = '<div class="detail-audio-scene">';
            if (s.tags_audio_scene) {
                sceneHtml += '<div class="scene-row">';
                sceneHtml += '<span class="scene-label">シーン:</span> ' + escapeHtml(s.tags_audio_scene);
                if (s.tags_audio_scene_spatial) {
                    sceneHtml += ' <span class="scene-label">空間:</span> ' + escapeHtml(s.tags_audio_scene_spatial);
                }
                if (s.tags_audio_scene_confidence) {
                    sceneHtml += ' <span style="color:#888">(' + Math.round(s.tags_audio_scene_confidence * 100) + '%)</span>';
                }
                sceneHtml += '</div>';
            }
            if (s.tags_audio_speaker_count) {
                var spkLabels = {monologue: '独り言', conversation: '会話', crowd: '群衆'};
                sceneHtml += '<div class="scene-row scene-sub">';
                sceneHtml += '<span class="scene-label">話者:</span> ' + escapeHtml(spkLabels[s.tags_audio_speaker_count] || s.tags_audio_speaker_count);
                sceneHtml += '</div>';
            }
            if (s.tags_audio_noise_sources) {
                var nsLabels = {vehicle_traffic:'交通',air_conditioning:'空調',mechanical_fan:'ファン',typing_keyboard:'タイピング',wind:'風',rain:'雨',construction:'工事',rustle:'衣擦れ',siren_emergency:'サイレン'};
                var nsList = s.tags_audio_noise_sources.split(',');
                sceneHtml += '<div class="scene-row scene-sub">';
                sceneHtml += '<span class="scene-label">ノイズ:</span> ';
                nsList.forEach(function(ns) {
                    sceneHtml += '<span class="tag-badge noise-source">' + escapeHtml(nsLabels[ns] || ns) + '</span> ';
                });
                sceneHtml += '</div>';
            }
            if (s.tags_audio_events) {
                var evLabels = {laughter:'笑い',crying:'泣き声',cough:'咳',sneeze:'くしゃみ',applause:'拍手・歓声',shout:'叫び',sigh:'ため息'};
                var evList = s.tags_audio_events.split(',');
                sceneHtml += '<div class="scene-row scene-sub">';
                sceneHtml += '<span class="scene-label">イベント:</span> ';
                evList.forEach(function(ev) {
                    sceneHtml += '<span class="tag-badge event-tag">' + escapeHtml(evLabels[ev] || ev) + '</span> ';
                });
                sceneHtml += '</div>';
            }
            if (s.tags_audio_music_detected) {
                sceneHtml += '<div class="scene-row scene-sub">';
                sceneHtml += '<span class="scene-label">音楽:</span> 検出あり';
                sceneHtml += '</div>';
            }
            sceneHtml += '</div>';
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
    var allFilterSelects = [filterConversationType, filterOutcome, filterDomain, filterImportanceV2, filterSpeaker, filterAudioScene];

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
        loadWhisperConfig();
        loadTerms();
        loadAutoDetected();
        loadSuggestions();
        loadSuggestionCount();

        loadHotwordsSettings();
        loadVoiceSettings();
        loadTaxonomySettings();
        loadRagSettings();
        loadRefinerSettings();
        checkSettingsAttention();
    }

    function closeSettings() {
        settingsModal.classList.add("hidden");
    }

    settingsBtn.addEventListener("click", openSettings);
    settingsClose.addEventListener("click", closeSettings);
    settingsModal.addEventListener("click", function (e) {
        if (e.target === settingsModal) closeSettings();
    });

    // --- Whisper config (read-only) ---

    var ENGINE_DESCRIPTIONS = {
        whisper: "faster-whisper large-v3 — 高精度・Hotwords デコーダバイアス対応 (~5 GB VRAM)",
        qwen3asr: "Qwen3-ASR 1.7B — 軽量・52言語対応・context boost (~3.4 GB VRAM)"
    };

    function loadWhisperConfig() {
        var loading = document.getElementById("whisper-loading");
        var info = document.getElementById("whisper-info");
        if (loading) loading.style.display = "";
        if (info) info.classList.add("hidden");
        fetch("/api/whisper/config").then(function (r) { return r.json(); }).then(function (d) {
            if (loading) loading.style.display = "none";
            if (info) info.classList.remove("hidden");
            var set = function (id, val) { var el = document.getElementById(id); if (el) el.textContent = val; };

            // Update engine selector
            var engineSelect = document.getElementById("engine-select");
            var engineDesc = document.getElementById("engine-desc");
            var currentEngine = d.engine || "whisper";
            if (engineSelect) engineSelect.value = currentEngine;
            if (engineDesc) engineDesc.textContent = ENGINE_DESCRIPTIONS[currentEngine] || "";

            set("whisper-engine", currentEngine);
            set("whisper-model-name", d.model_name || "-");
            set("whisper-compute-type", d.compute_type || "-");
            var statusEl = document.getElementById("whisper-loaded");
            if (statusEl) {
                statusEl.textContent = d.loaded ? "Loaded" : "Unloaded";
                statusEl.className = d.loaded ? "status-loaded" : "status-unloaded";
            }
            set("whisper-session-count", d.session_count != null ? d.session_count : "-");
            set("whisper-language", d.language || "-");
            set("whisper-beam-size", d.beam_size != null ? d.beam_size : "-");
            set("whisper-vad-filter", d.vad_filter ? "Enabled" : "Disabled");
            set("whisper-ngram", d.no_repeat_ngram_size != null ? d.no_repeat_ngram_size : "-");
            set("whisper-patience", d.patience != null ? d.patience : "-");
            set("whisper-logprob", d.log_prob_threshold != null ? d.log_prob_threshold : "-");
            set("whisper-seg-max", d.segment_max_duration != null ? d.segment_max_duration + "s" : "-");
            set("whisper-normalize", d.normalize_enabled ? "Enabled" : "Disabled");
            set("whisper-noise-reduce", d.noise_reduce_enabled ? "Enabled" : "Disabled");

            // Show/hide Whisper-specific rows based on engine
            var whisperOnlyRows = ["whisper-compute-type", "whisper-beam-size",
                "whisper-vad-filter", "whisper-ngram", "whisper-patience", "whisper-logprob"];
            whisperOnlyRows.forEach(function (id) {
                var el = document.getElementById(id);
                if (el && el.parentElement) {
                    el.parentElement.style.display = (currentEngine === "whisper") ? "" : "none";
                }
            });
        }).catch(function () {
            if (loading) loading.textContent = "ASR 設定を取得できませんでした";
        });
    }

    // Engine switch button
    var engineSwitchBtn = document.getElementById("engine-switch-btn");
    if (engineSwitchBtn) {
        engineSwitchBtn.addEventListener("click", function () {
            var select = document.getElementById("engine-select");
            var status = document.getElementById("engine-switch-status");
            if (!select) return;
            var newEngine = select.value;

            if (status) { status.textContent = "切り替え中..."; status.style.color = ""; }
            engineSwitchBtn.disabled = true;

            fetch("/api/engine/switch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ engine: newEngine }),
            })
            .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
            .then(function (result) {
                engineSwitchBtn.disabled = false;
                if (result.ok) {
                    if (status) {
                        status.textContent = result.data.message || "切り替え完了";
                        status.style.color = "#4caf50";
                    }
                    // Reload config to reflect new engine
                    setTimeout(loadWhisperConfig, 500);
                } else {
                    if (status) {
                        status.textContent = result.data.error || "切り替え失敗";
                        status.style.color = "#f44336";
                    }
                }
                setTimeout(function () { if (status) status.textContent = ""; }, 4000);
            })
            .catch(function () {
                engineSwitchBtn.disabled = false;
                if (status) { status.textContent = "接続エラー"; status.style.color = "#f44336"; }
                setTimeout(function () { if (status) status.textContent = ""; }, 4000);
            });
        });
    }

    // Update description on engine select change
    var engineSelect = document.getElementById("engine-select");
    if (engineSelect) {
        engineSelect.addEventListener("change", function () {
            var desc = document.getElementById("engine-desc");
            if (desc) desc.textContent = ENGINE_DESCRIPTIONS[this.value] || "";
        });
    }

    // Settings left nav tabs
    var settingsNavItems = document.querySelectorAll(".settings-nav-item");
    var settingsTabContents = document.querySelectorAll(".settings-content > .tab-content");
    for (var i = 0; i < settingsNavItems.length; i++) {
        settingsNavItems[i].addEventListener("click", function () {
            var targetId = this.getAttribute("data-tab");
            settingsNavItems.forEach(function (n) { n.classList.remove("active"); });
            settingsTabContents.forEach(function (c) { c.classList.remove("active"); });
            this.classList.add("active");
            var target = document.getElementById(targetId);
            if (target) target.classList.add("active");
        });
    }

    // Mobile: category accordion
    var navCategories = document.querySelectorAll(".settings-nav-category");
    for (var ci = 0; ci < navCategories.length; ci++) {
        navCategories[ci].addEventListener("click", function () {
            var isExpanded = this.classList.contains("expanded");
            // Collapse all
            navCategories.forEach(function (c) { c.classList.remove("expanded"); });
            document.querySelectorAll(".settings-nav-item").forEach(function (it) { it.classList.remove("mobile-visible"); });
            if (!isExpanded) {
                this.classList.add("expanded");
                var sib = this.nextElementSibling;
                while (sib && sib.classList.contains("settings-nav-item")) {
                    sib.classList.add("mobile-visible");
                    sib = sib.nextElementSibling;
                }
            }
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
            renderSuggestionsInto(suggestionsListEl, data, function () {
                loadSuggestions();
            });
            // Update badge
            if (data.length > 0) {
                suggestionBadge.textContent = data.length;
                suggestionBadge.classList.remove("hidden");
            } else {
                suggestionBadge.classList.add("hidden");
            }
        });
    }

    function renderSuggestionsInto(container, list, onChanged) {
        if (list.length === 0) {
            container.innerHTML = '<div class="loading">No pending suggestions</div>';
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
        container.innerHTML = html;

        // Bind events
        var approveBtns = container.querySelectorAll(".suggestion-approve");
        for (var i = 0; i < approveBtns.length; i++) {
            approveBtns[i].addEventListener("click", function () {
                var sid = this.getAttribute("data-id");
                fetch("/api/suggestions/" + sid + "/approve", { method: "POST" })
                    .then(function (r) { return r.json(); })
                    .then(function () { termsCache = null; if (onChanged) onChanged(); loadTerms(); loadAutoDetected(); });
            });
        }

        var rejectBtns = container.querySelectorAll(".suggestion-reject");
        for (var j = 0; j < rejectBtns.length; j++) {
            rejectBtns[j].addEventListener("click", function () {
                var sid = this.getAttribute("data-id");
                fetch("/api/suggestions/" + sid + "/reject", { method: "POST" })
                    .then(function (r) { return r.json(); })
                    .then(function () { if (onChanged) onChanged(); });
            });
        }

        var editBtns = container.querySelectorAll(".suggestion-edit-btn");
        for (var k = 0; k < editBtns.length; k++) {
            editBtns[k].addEventListener("click", function () {
                var sid = this.getAttribute("data-id");
                editSuggestionInContainer(container, sid, onChanged);
            });
        }
    }

    function editSuggestionInContainer(container, id, onChanged) {
        var correctSpan = container.querySelector('.correct[data-id="' + id + '"]');
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
                if (onChanged) onChanged();
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

    // Global keyboard shortcuts
    document.addEventListener("keydown", function (e) {
        // Skip shortcuts when typing in inputs
        var tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

        // Review mode shortcuts
        if (reviewState.active) {
            if (e.key === "Escape") {
                if (reviewState.drawerOpen) {
                    toggleReviewDrawer(false);
                } else if (reviewState.editing) {
                    reviewState.editing = false;
                    loadNextReviewItem();
                } else {
                    closeReviewMode();
                }
                return;
            }
            if (e.key === "d" || e.key === "D") { toggleReviewDrawer(); return; }
            if (e.key === "a" || e.key === "A") { reviewAccept(); return; }
            if (e.key === "e" || e.key === "E") { reviewEdit(); return; }
            if (e.key === "s" || e.key === "S" || e.key === " ") { e.preventDefault(); reviewSkip(); return; }
            return;
        }

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


    var hotwordsSettingsCache = null;
    var hotwordsStatusCache = null;

    function loadHotwordsStatus() {
        fetch("/api/hotwords/status").then(function (r) {
            if (!r.ok) return null;
            return r.json();
        }).then(function (data) {
            if (!data) return;
            hotwordsStatusCache = data;
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
        var maxPerClass = document.getElementById("export-max-per-class");
        if (maxPerClass && parseInt(maxPerClass.value, 10) > 0) {
            params += "&max_per_class=" + encodeURIComponent(maxPerClass.value);
        }
        var minClassSamples = document.getElementById("export-min-class-samples");
        if (minClassSamples && parseInt(minClassSamples.value, 10) > 0) {
            params += "&min_class_samples=" + encodeURIComponent(minClassSamples.value);
        }
        var valRatio = document.getElementById("export-val-ratio");
        if (valRatio && parseFloat(valRatio.value) > 0) {
            params += "&val_ratio=" + encodeURIComponent(valRatio.value);
        }
        // KC-targeted filters (new in V2 KC-sourced export)
        var reqTagged = document.getElementById("export-require-tagged");
        if (reqTagged && reqTagged.checked) params += "&require_tagged=true";
        var reqSummary = document.getElementById("export-require-summary");
        if (reqSummary && reqSummary.checked) params += "&require_summary=true";
        var excPersonal = document.getElementById("export-exclude-personal-only");
        if (excPersonal && excPersonal.checked) params += "&exclude_personal_only=true";
        var highValOr = document.getElementById("export-high-value-or");
        if (highValOr && highValOr.checked) params += "&high_value_or=true";
        var extAudio = document.getElementById("export-external-audio-max");
        if (extAudio && extAudio.value !== "" && !isNaN(parseFloat(extAudio.value))) {
            params += "&external_audio_max=" + encodeURIComponent(extAudio.value);
        }
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
                    var previewHtml =
                        '<div class="export-preview-line"><strong>' + data.sessions + '</strong> sessions / <strong>' + data.segments + '</strong> segments</div>' +
                        '<div class="export-preview-line">Date: ' + escapeHtml(data.date_from || "N/A") + ' ~ ' + escapeHtml(data.date_to || "N/A") + '</div>' +
                        '<div class="export-preview-line">Est. size: <strong>' + formatBytes(data.est_size_bytes) + '</strong></div>';
                    if (data.balanced_sessions != null) {
                        previewHtml += '<div class="export-preview-line" style="margin-top:6px;"><strong>均等化後: ' + data.balanced_sessions + '</strong> sessions</div>';
                        var bd = data.class_breakdown || {};
                        var bdKeys = Object.keys(bd).sort(function (a, b) { return bd[b].total - bd[a].total; });
                        previewHtml += '<div class="export-class-breakdown">';
                        bdKeys.forEach(function (cls) {
                            var info = bd[cls];
                            var capped = info.sampled < info.total;
                            var excluded = info.excluded;
                            var label = escapeHtml(cls) + ': ' + info.sampled + '/' + info.total;
                            if (excluded) label += ' <span style="color:#e74c3c;">(除外: 件数不足)</span>';
                            else if (capped) label += ' <span style="color:#e67e22;">(capped)</span>';
                            previewHtml += '<div class="text-dim" style="font-size:0.75rem;">' + label + '</div>';
                        });
                        previewHtml += '</div>';
                    }
                    if (data.train_sessions != null) {
                        previewHtml += '<div class="export-preview-line" style="margin-top:4px;">Train: <strong>' + data.train_sessions + '</strong> / Val: <strong>' + data.val_sessions + '</strong></div>';
                    }
                    exportPreviewContent.innerHTML = previewHtml;
                    var effectiveSessions = data.balanced_sessions != null ? data.balanced_sessions : data.sessions;
                    exportDownloadBtn.disabled = effectiveSessions === 0;
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
    // Succession RAG (事業相談) tab — multi-turn chat
    // ============================================================

    var successionStatusBar = document.getElementById("succession-status-bar");
    var successionRebuildBtn = document.getElementById("succession-rebuild-btn");
    var successionReloadWikiBtn = document.getElementById("succession-reload-wiki-btn");
    var successionClearBtn = document.getElementById("succession-clear-btn");
    var successionRebuildStatus = document.getElementById("succession-rebuild-status");
    var successionQuery = document.getElementById("succession-query");
    var successionK = document.getElementById("succession-k");
    var successionAskBtn = document.getElementById("succession-ask-btn");
    var successionAskStatus = document.getElementById("succession-ask-status");
    var successionChatHistory = document.getElementById("succession-chat-history");
    var successionChatEmpty = document.getElementById("succession-chat-empty");
    var successionTurnCounter = document.getElementById("succession-turn-counter");

    // In-memory conversation state. Server is stateless; we send the full
    // history with each /ask call, and the server caps it to MAX_HISTORY_TURNS.
    // Each entry: { role: "user"|"assistant", content: "...", meta?: {...},
    //                wiki_topics?: [...], sources?: [...] }
    var successionConversation = [];
    var SUCCESSION_HISTORY_CAP = 10;

    function updateTurnCounter() {
        if (!successionTurnCounter) return;
        var n = successionConversation.length;
        successionTurnCounter.textContent = n > 0
            ? "会話: " + n + " ターン (cap " + SUCCESSION_HISTORY_CAP + ")"
            : "";
    }

    function refreshSuccessionStatus() {
        if (!successionStatusBar) return;
        fetch("/api/succession/status")
            .then(function (r) { return r.json(); })
            .then(function (s) {
                if (s.error) {
                    successionStatusBar.textContent = "Error: " + s.error;
                    return;
                }
                if (!s.enabled) {
                    successionStatusBar.textContent = "無効 (GEMINI_API_KEY / OPENAI_API_KEY 両方必要)";
                    successionRebuildBtn.disabled = true;
                    if (successionReloadWikiBtn) successionReloadWikiBtn.disabled = true;
                    successionAskBtn.disabled = true;
                    return;
                }
                var mode = s.retrieval_mode || "?";
                var wikiText = "";
                if (s.wiki && s.wiki.enabled) {
                    wikiText = " | wiki=" + s.wiki.topic_count + " topics";
                }
                successionStatusBar.textContent =
                    "Mode: " + mode +
                    " | KC: " + s.index_size + wikiText +
                    " | " + (s.ready ? "✓ Ready" : "未構築") +
                    " | answer=" + (s.answer_model || "?");
                successionAskBtn.disabled = !s.ready;
            })
            .catch(function (e) {
                successionStatusBar.textContent = "Error: " + e.message;
            });
    }

    function renderChatHistory() {
        if (!successionChatHistory) return;
        if (successionConversation.length === 0) {
            successionChatHistory.innerHTML =
                '<div id="succession-chat-empty" class="text-dim" ' +
                'style="text-align:center;padding:40px 0;font-size:0.85rem;">' +
                '質問を入力すると、ここに会話履歴が表示されます。</div>';
            updateTurnCounter();
            return;
        }
        var html = "";
        successionConversation.forEach(function (turn, idx) {
            var isUser = turn.role === "user";
            var bg = isUser ? "#1e3a5f" : "#1f4d2e";
            var label = isUser ? "👤 You" : "🤖 Assistant";
            html +=
                '<div style="margin-bottom:14px;padding:10px;border-radius:6px;background:' + bg + ';">' +
                '<div class="text-dim" style="font-size:0.7rem;margin-bottom:4px;">' +
                escapeHtml(label) + ' (#' + (idx + 1) + ')</div>' +
                '<div style="white-space:pre-wrap;line-height:1.5;font-size:0.85rem;">' +
                escapeHtml(turn.content) + '</div>';
            if (!isUser && turn.meta) {
                html += '<div class="text-dim" style="font-size:0.65rem;margin-top:6px;">' +
                    escapeHtml(turn.meta) + '</div>';
            }
            if (!isUser && (turn.wiki_topics || turn.sources)) {
                html += '<details style="margin-top:6px;">' +
                    '<summary style="cursor:pointer;font-size:0.75rem;">参照した素材</summary>' +
                    '<div style="margin-top:6px;font-size:0.75rem;">';
                if (turn.wiki_topics && turn.wiki_topics.length) {
                    turn.wiki_topics.forEach(function (t, i) {
                        html += '<div style="margin:6px 0;padding:6px;border-left:3px solid #4a9;">' +
                            '<strong>[Wiki #' + (i + 1) + '] ' + escapeHtml(t.title) + '</strong>' +
                            ' <span class="text-dim">(' + escapeHtml(t.domain) +
                            ', score=' + (t.aggregate_score || 0).toFixed(3) +
                            ', kcs=' + (t.matched_kc_count || 0) + ')</span>' +
                            '</div>';
                    });
                }
                if (turn.sources && turn.sources.length) {
                    turn.sources.slice(0, 5).forEach(function (s, i) {
                        html += '<div style="margin:4px 0 4px 12px;padding:4px;border-left:2px solid #888;">' +
                            '<div class="text-dim" style="font-size:0.65rem;">[KC #' + (i + 1) + '] ' +
                            escapeHtml(s.session_id) + ' (rrf=' + (s.rrf_score || 0).toFixed(3) + ')</div>' +
                            '<div><strong>Q:</strong> ' + escapeHtml(s.question.substring(0, 100)) + '</div>' +
                            '</div>';
                    });
                }
                html += '</div></details>';
            }
            html += '</div>';
        });
        successionChatHistory.innerHTML = html;
        // Auto-scroll to bottom so latest turn is visible
        successionChatHistory.scrollTop = successionChatHistory.scrollHeight;
        updateTurnCounter();
    }

    function clearConversation() {
        successionConversation = [];
        renderChatHistory();
    }

    if (successionRebuildBtn) {
        successionRebuildBtn.addEventListener("click", function () {
            successionRebuildBtn.disabled = true;
            successionRebuildStatus.textContent = "KC index 構築中... (数十秒〜数分)";
            successionRebuildStatus.style.color = "";
            fetch("/api/succession/rebuild", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ force: false }),
            })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.error) {
                        successionRebuildStatus.textContent = "Error: " + data.error;
                        successionRebuildStatus.style.color = "#e74c3c";
                        return;
                    }
                    successionRebuildStatus.textContent =
                        "KC: " + data.status + " (" + (data.size || 0) + " KCs)";
                    successionRebuildStatus.style.color = "#27ae60";
                    refreshSuccessionStatus();
                })
                .catch(function (e) {
                    successionRebuildStatus.textContent = "Error: " + e.message;
                    successionRebuildStatus.style.color = "#e74c3c";
                })
                .finally(function () {
                    successionRebuildBtn.disabled = false;
                });
        });
    }

    if (successionReloadWikiBtn) {
        successionReloadWikiBtn.addEventListener("click", function () {
            successionReloadWikiBtn.disabled = true;
            successionRebuildStatus.textContent = "Wiki 再ロード中...";
            successionRebuildStatus.style.color = "";
            fetch("/api/succession/reload_wiki", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({}),
            })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.error) {
                        successionRebuildStatus.textContent = "Error: " + data.error;
                        successionRebuildStatus.style.color = "#e74c3c";
                        return;
                    }
                    successionRebuildStatus.textContent =
                        "Wiki: " + data.topic_count + " topics, " +
                        data.session_count + " sessions";
                    successionRebuildStatus.style.color = "#27ae60";
                    refreshSuccessionStatus();
                })
                .catch(function (e) {
                    successionRebuildStatus.textContent = "Error: " + e.message;
                    successionRebuildStatus.style.color = "#e74c3c";
                })
                .finally(function () {
                    successionReloadWikiBtn.disabled = false;
                });
        });
    }

    if (successionClearBtn) {
        successionClearBtn.addEventListener("click", clearConversation);
    }

    function sendSuccessionQuery() {
        var q = (successionQuery.value || "").trim();
        if (!q) return;
        var k = parseInt(successionK.value, 10) || 5;

        // Snapshot prior history (server caps internally too)
        var historyPayload = successionConversation.map(function (t) {
            return { role: t.role, content: t.content };
        });

        // Optimistically add the user turn so the user sees their question
        successionConversation.push({ role: "user", content: q });
        successionQuery.value = "";
        renderChatHistory();

        successionAskBtn.disabled = true;
        successionAskStatus.textContent = "回答中...";
        successionAskStatus.style.color = "";

        fetch("/api/succession/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: q, k: k, history: historyPayload }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) {
                    successionAskStatus.textContent = "Error: " + data.error;
                    successionAskStatus.style.color = "#e74c3c";
                    // Roll back the optimistic user message so retry works
                    successionConversation.pop();
                    renderChatHistory();
                    return;
                }
                successionAskStatus.textContent = "";
                var meta = "model=" + (data.model || "?") +
                    " | " + (data.elapsed_sec ? data.elapsed_sec.toFixed(2) + "s" : "") +
                    " | mode=" + (data.retrieval_mode || "?") +
                    (data.warnings && data.warnings.length
                        ? " | warn=" + data.warnings.join(",")
                        : "");
                successionConversation.push({
                    role: "assistant",
                    content: data.answer || "(no answer)",
                    meta: meta,
                    wiki_topics: data.wiki_topics || [],
                    sources: data.sources || [],
                });
                renderChatHistory();
            })
            .catch(function (e) {
                successionAskStatus.textContent = "Error: " + e.message;
                successionAskStatus.style.color = "#e74c3c";
                successionConversation.pop();
                renderChatHistory();
            })
            .finally(function () {
                successionAskBtn.disabled = false;
            });
    }

    if (successionAskBtn) {
        successionAskBtn.addEventListener("click", sendSuccessionQuery);
    }

    if (successionQuery) {
        // Ctrl/Cmd+Enter to submit
        successionQuery.addEventListener("keydown", function (e) {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                e.preventDefault();
                sendSuccessionQuery();
            }
        });
    }

    // Initial status load + empty render
    refreshSuccessionStatus();
    renderChatHistory();

    // ============================================================
    // Taxonomy Settings tab
    // ============================================================

    var taxonomyEditCache = null;
    var taxonomyLabelCounts = {};  // { axis: { label: count } }

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
            // Fetch label counts in parallel (non-blocking)
            fetch("/api/taxonomy-label-counts").then(function (r) { return r.json(); })
                .then(function (counts) { taxonomyLabelCounts = counts; renderTaxonomySettings(); })
                .catch(function () {});
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
            var axisCounts = (taxonomyLabelCounts || {})[axis.key] || {};
            var html = "";
            items.forEach(function (item, idx) {
                var cnt = axisCounts[item] || 0;
                var cntClass = cnt === 0 ? "taxonomy-cnt-zero" : "taxonomy-cnt";
                html += '<div class="taxonomy-item">';
                html += '<span class="taxonomy-item-text">' + escapeHtml(item) + '</span>';
                html += '<span class="' + cntClass + '">' + cnt + '</span>';
                html += '<button class="btn-sm btn-xs taxonomy-edit" data-axis="' + axis.key + '" data-idx="' + idx + '">Edit</button>';
                html += '<button class="btn-danger btn-xs taxonomy-del" data-axis="' + axis.key + '" data-idx="' + idx + '">Del</button>';
                html += '</div>';
            });
            // Warning if item count > 20
            if (items.length > 20) {
                html += '<div class="taxonomy-warn">項目数が多すぎます (' + items.length + ')。ファインチューニング精度のため 20 以下を推奨。</div>';
            }
            listEl.innerHTML = html || '<div class="text-dim">項目がありません</div>';

            // Edit handlers
            var editBtns = listEl.querySelectorAll(".taxonomy-edit");
            for (var ei = 0; ei < editBtns.length; ei++) {
                editBtns[ei].addEventListener("click", function () {
                    var ak = this.getAttribute("data-axis");
                    var ai = parseInt(this.getAttribute("data-idx"), 10);
                    var itemDiv = this.closest(".taxonomy-item");
                    var textSpan = itemDiv.querySelector(".taxonomy-item-text");
                    var currentVal = taxonomyEditCache[ak][ai];

                    // Replace text and buttons with input + OK/Cancel
                    itemDiv.innerHTML =
                        '<input type="text" class="taxonomy-item-edit-input" value="' + escapeHtml(currentVal) + '">' +
                        '<button class="btn-primary btn-xs taxonomy-edit-ok" data-axis="' + ak + '" data-idx="' + ai + '">OK</button>' +
                        '<button class="btn-sm btn-xs taxonomy-edit-cancel">Cancel</button>';

                    var input = itemDiv.querySelector(".taxonomy-item-edit-input");
                    input.focus();
                    input.select();

                    function applyEdit() {
                        var newVal = input.value.trim();
                        if (!newVal || newVal === currentVal) {
                            renderTaxonomySettings();
                            return;
                        }
                        // Check for duplicates
                        if (taxonomyEditCache[ak].indexOf(newVal) !== -1) {
                            input.style.borderColor = "#e74c3c";
                            return;
                        }
                        var oldVal = currentVal;
                        taxonomyEditCache[ak][ai] = newVal;

                        // Build rename mapping for this edit
                        var renames = {};
                        renames[ak] = {};
                        renames[ak][oldVal] = newVal;

                        // If domain renamed, update keyword_anchors and domain_descriptions keys
                        if (ak === "domains") {
                            if (taxonomyEditCache.keyword_anchors && taxonomyEditCache.keyword_anchors[oldVal]) {
                                taxonomyEditCache.keyword_anchors[newVal] = taxonomyEditCache.keyword_anchors[oldVal];
                                delete taxonomyEditCache.keyword_anchors[oldVal];
                            }
                            if (taxonomyEditCache.domain_descriptions && taxonomyEditCache.domain_descriptions[oldVal]) {
                                taxonomyEditCache.domain_descriptions[newVal] = taxonomyEditCache.domain_descriptions[oldVal];
                                delete taxonomyEditCache.domain_descriptions[oldVal];
                            }
                        }

                        // Save immediately to server
                        renderTaxonomySettings();
                        saveTaxonomyNow(oldVal + " → " + newVal, renames);
                    }

                    itemDiv.querySelector(".taxonomy-edit-ok").addEventListener("click", applyEdit);
                    itemDiv.querySelector(".taxonomy-edit-cancel").addEventListener("click", function () {
                        renderTaxonomySettings();
                    });
                    input.addEventListener("keydown", function (e) {
                        if (e.key === "Enter") applyEdit();
                        if (e.key === "Escape") renderTaxonomySettings();
                    });
                });
            }

            var delBtns = listEl.querySelectorAll(".taxonomy-del");
            for (var i = 0; i < delBtns.length; i++) {
                delBtns[i].addEventListener("click", function () {
                    var ak = this.getAttribute("data-axis");
                    var ai = parseInt(this.getAttribute("data-idx"), 10);
                    var removed = taxonomyEditCache[ak][ai];
                    taxonomyEditCache[ak].splice(ai, 1);
                    // If a domain was removed, also remove its keyword_anchors entry
                    if (ak === "domains" && taxonomyEditCache.keyword_anchors) {
                        var removedDomain = Object.keys(taxonomyEditCache.keyword_anchors).filter(function (d) {
                            return (taxonomyEditCache.domains || []).indexOf(d) === -1;
                        });
                        removedDomain.forEach(function (d) { delete taxonomyEditCache.keyword_anchors[d]; });
                    }
                    renderTaxonomySettings();
                    saveTaxonomyNow(removed + " を削除");
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
                var removedKw = (taxonomyEditCache.keyword_anchors[d] || [])[idx];
                if (taxonomyEditCache.keyword_anchors[d]) {
                    taxonomyEditCache.keyword_anchors[d].splice(idx, 1);
                }
                renderKeywordAnchors();
                saveTaxonomyNow(d + ": " + removedKw + " を削除");
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
                saveTaxonomyNow(d + ": " + val + " を追加");
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
            saveTaxonomyNow(val + " を追加");
        });
    }

    // Common save function for all taxonomy mutations
    var taxonomyStatusEl = document.getElementById("taxonomy-status");
    function saveTaxonomyNow(statusMsg, renames) {
        if (!taxonomyEditCache) return;
        if (taxonomyStatusEl) {
            taxonomyStatusEl.textContent = "保存中...";
            taxonomyStatusEl.style.color = "";
        }
        var payload = { taxonomy: taxonomyEditCache };
        if (renames) payload.renames = renames;
        fetch("/api/tag-taxonomy", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        }).then(function (r) { return r.json(); }).then(function (data) {
            if (data.error) {
                if (taxonomyStatusEl) {
                    taxonomyStatusEl.textContent = "Error: " + data.error;
                    taxonomyStatusEl.style.color = "#e74c3c";
                }
                return;
            }
            var renamed = data.renamed || 0;
            var msg = statusMsg || "保存しました";
            if (renamed > 0) msg += " (" + renamed + " セッション更新)";
            if (taxonomyStatusEl) {
                taxonomyStatusEl.textContent = msg;
                taxonomyStatusEl.style.color = "#2ecc71";
                setTimeout(function () { taxonomyStatusEl.textContent = ""; }, 5000);
            }
            loadTaxonomy();
            loadPage(currentPage);
            // Impact: truly deleted values
            if (data.impact && Object.keys(data.impact).length > 0) {
                var impactMsg = "以下のタグ値が使用中のセッションから削除されました:\n";
                for (var k in data.impact) {
                    impactMsg += "  " + k + ": " + data.impact[k] + " セッション\n";
                }
                impactMsg += "\nこれらのセッションを新しいタクソノミーで再タグ付けしますか？";
                if (confirm(impactMsg)) {
                    fetch("/api/session-tags/start", { method: "POST" })
                        .then(function (r) { return r.json(); })
                        .then(function () {
                            if (taxonomyStatusEl) taxonomyStatusEl.textContent = "再タグ付け開始";
                            showBatchPanel("tags", { running: true, total: 0, processed: 0 });
                            startBatchPolling("tags");
                        });
                }
            }
        }).catch(function (err) {
            if (taxonomyStatusEl) {
                taxonomyStatusEl.textContent = "Error: " + err.message;
                taxonomyStatusEl.style.color = "#e74c3c";
            }
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
                var typeColors = {"split": "#3498db", "merge": "#e67e22", "new": "#2ecc71", "remove": "#e74c3c"};
                data.pending.changes.forEach(function (c, idx) {
                    var typeColor = typeColors[c.type] || "#aaa";
                    html += "<div class='refiner-change-card accepted' data-index='" + idx + "'>";
                    html += "<div class='card-header'>";
                    html += "<input type='checkbox' class='refiner-change-cb' data-index='" + idx + "' checked>";
                    html += "<span style='color:" + typeColor + ";font-weight:bold;'>[" + c.type + "]</span> ";
                    html += "<span>" + (c.axis || "") + "</span>";
                    html += "</div>";
                    html += "<div class='card-detail'>" + (c.description || "") + "</div>";
                    if (c.before && c.before.length) {
                        html += "<div class='card-detail' style='color:#999;'>Before: " + c.before.join(", ") + "</div>";
                    }
                    if (c.after && c.after.length) {
                        html += "<div class='card-detail' style='color:#ccc;'>After: " + c.after.join(", ") + "</div>";
                    }
                    html += "</div>";
                });
                changesList.innerHTML = html || "<div class='text-dim'>変更なし</div>";

                // Toggle checkbox on card click
                changesList.querySelectorAll(".refiner-change-card").forEach(function (card) {
                    card.addEventListener("click", function (e) {
                        if (e.target.type === "checkbox") return; // let checkbox handle itself
                        var cb = card.querySelector(".refiner-change-cb");
                        if (cb) { cb.checked = !cb.checked; cb.dispatchEvent(new Event("change")); }
                    });
                });
                changesList.querySelectorAll(".refiner-change-cb").forEach(function (cb) {
                    cb.addEventListener("change", function () {
                        var card = cb.closest(".refiner-change-card");
                        if (card) {
                            card.classList.toggle("accepted", cb.checked);
                            card.classList.toggle("rejected", !cb.checked);
                        }
                    });
                });
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

    function refinerApproveWith(indices) {
        var body = indices !== null ? JSON.stringify({ accepted_indices: indices }) : "{}";
        var label = indices !== null
            ? (indices.length + " 件の変更を適用")
            : "全ての変更を適用";
        if (!confirm(label + "してタクソノミーを更新しますか？")) return;
        fetch("/api/taxonomy/refine/approve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: body
        }).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function (data) {
            var msg = "承認: " + (data.accepted || 0) + " 件適用";
            if (data.rejected) msg += "、" + data.rejected + " 件スキップ";
            alert(msg + "。タクソノミーが更新されました。");
            loadRefinerSettings();
            loadTaxonomySettings();
        }).catch(function (err) {
            alert("Error: " + err.message);
        });
    }

    var refinerApproveBtn = document.getElementById("refiner-approve-btn");
    if (refinerApproveBtn) {
        refinerApproveBtn.addEventListener("click", function () {
            var checked = document.querySelectorAll(".refiner-change-cb:checked");
            var indices = [];
            checked.forEach(function (cb) { indices.push(parseInt(cb.getAttribute("data-index"), 10)); });
            if (indices.length === 0) { alert("適用する変更を選択してください。"); return; }
            refinerApproveWith(indices);
        });
    }

    var refinerApproveAllBtn = document.getElementById("refiner-approve-all-btn");
    if (refinerApproveAllBtn) {
        refinerApproveAllBtn.addEventListener("click", function () {
            refinerApproveWith(null);
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

    // ============================================================
    // Review Queue Mode
    // ============================================================

    var reviewOverlay = document.getElementById("review-overlay");
    var reviewContent = document.getElementById("review-content");
    var reviewProgressFill = document.getElementById("review-progress-fill");
    var reviewProgressText = document.getElementById("review-progress-text");
    var reviewStatsText = document.getElementById("review-stats-text");
    var reviewBtn = document.getElementById("review-btn");
    var reviewCountBadge = document.getElementById("review-count-badge");
    var reviewCloseBtn = document.getElementById("review-close-btn");
    var reviewAcceptBtn = document.getElementById("review-accept-btn");
    var reviewEditBtn = document.getElementById("review-edit-btn");
    var reviewSkipBtn = document.getElementById("review-skip-btn");

    var reviewState = {
        active: false,
        currentItem: null,
        totalPending: 0,
        reviewedCount: 0,
        skippedOffset: 0,
        editing: false,
    };

    function loadReviewStats() {
        fetch("/api/review/stats").then(function (r) { return r.json(); }).then(function (data) {
            reviewState.totalPending = data.total_pending || 0;
            if (reviewCountBadge) {
                if (reviewState.totalPending > 0) {
                    reviewCountBadge.textContent = reviewState.totalPending;
                    reviewCountBadge.classList.remove("hidden");
                } else {
                    reviewCountBadge.classList.add("hidden");
                }
            }
        }).catch(function () {});
    }

    function openReviewMode() {
        reviewState.active = true;
        reviewState.reviewedCount = 0;
        reviewState.skippedOffset = 0;
        reviewState.editing = false;
        reviewOverlay.classList.remove("hidden");
        document.body.style.overflow = "hidden";
        loadNextReviewItem();
    }

    function closeReviewMode() {
        reviewState.active = false;
        reviewState.currentItem = null;
        reviewState.editing = false;
        reviewState.drawerOpen = false;
        var drawer = document.getElementById("review-drawer");
        if (drawer) drawer.classList.remove("visible");
        reviewOverlay.classList.add("hidden");
        document.body.style.overflow = "";
        loadReviewStats();
        loadPage(currentPage);
    }

    // --- Review side drawer ---

    reviewState.drawerOpen = false;

    var reviewDrawer = document.getElementById("review-drawer");
    var reviewDrawerBtn = document.getElementById("review-drawer-btn");
    var drawerCloseBtn = document.getElementById("drawer-close-btn");
    var drawerSuggestionsEl = document.getElementById("drawer-suggestions-list");
    var drawerTabSuggestionsBadge = document.getElementById("drawer-tab-suggestions-badge");
    var drawerTabRefinerBadge = document.getElementById("drawer-tab-refiner-badge");

    function toggleReviewDrawer(forceState) {
        var open = forceState != null ? forceState : !reviewState.drawerOpen;
        reviewState.drawerOpen = open;
        if (reviewDrawer) {
            if (open) {
                reviewDrawer.classList.add("visible");
                loadDrawerSuggestions();
                loadDrawerRefiner();
            } else {
                reviewDrawer.classList.remove("visible");
            }
        }
    }

    function loadDrawerSuggestions() {
        if (!drawerSuggestionsEl) return;
        fetch("/api/suggestions").then(function (r) { return r.json(); }).then(function (data) {
            renderSuggestionsInto(drawerSuggestionsEl, data, function () {
                loadDrawerSuggestions();
                // Invalidate settings modal cache
                suggestionsListEl.innerHTML = "";
            });
            // Update drawer tab badge
            if (drawerTabSuggestionsBadge) {
                if (data && data.length > 0) {
                    drawerTabSuggestionsBadge.textContent = data.length;
                    drawerTabSuggestionsBadge.classList.remove("hidden");
                } else {
                    drawerTabSuggestionsBadge.classList.add("hidden");
                }
            }
        });
    }

    // --- Drawer Refiner ---
    var drawerRefinerStatus = document.getElementById("drawer-refiner-status");
    var drawerRefinerPending = document.getElementById("drawer-refiner-pending");

    function loadDrawerRefiner() {
        if (!drawerRefinerStatus) return;
        drawerRefinerStatus.textContent = "読み込み中...";
        drawerRefinerStatus.classList.remove("hidden");
        if (drawerRefinerPending) drawerRefinerPending.classList.add("hidden");

        fetch("/api/taxonomy/refine/status").then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function (data) {
            var statusParts = [];
            statusParts.push(data.enabled ? "有効" : "無効");
            if (data.running) statusParts.push("実行中 (" + (data.phase || "") + ")");
            if (data.last_run) statusParts.push("最終: " + data.last_run);
            drawerRefinerStatus.textContent = statusParts.join(" / ");

            // Update drawer tab badge
            if (drawerTabRefinerBadge) {
                if (data.has_pending) {
                    drawerTabRefinerBadge.textContent = "!";
                    drawerTabRefinerBadge.classList.remove("hidden");
                } else {
                    drawerTabRefinerBadge.classList.add("hidden");
                }
            }

            if (data.has_pending) {
                loadDrawerRefinerPending();
            }
            if (data.running) {
                setTimeout(loadDrawerRefiner, 5000);
            }
        }).catch(function (err) {
            drawerRefinerStatus.textContent = "Error: " + err.message;
        });
    }

    function loadDrawerRefinerPending() {
        if (!drawerRefinerPending) return;
        fetch("/api/taxonomy/refine/pending").then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function (data) {
            if (!data.pending) return;
            drawerRefinerPending.classList.remove("hidden");

            var changes = data.pending.changes || [];
            var typeColors = {"split": "#3498db", "merge": "#e67e22", "new": "#2ecc71", "remove": "#e74c3c"};
            var html = "<div style='margin-bottom:8px;font-size:0.85em;color:var(--text-dim);'>" + changes.length + " 件の提案</div>";

            changes.forEach(function (c, idx) {
                var typeColor = typeColors[c.type] || "#aaa";
                html += "<div class='refiner-change-card accepted' data-index='" + idx + "' style='margin-bottom:6px;padding:8px;'>";
                html += "<div style='margin-bottom:4px;'>";
                html += "<span style='color:" + typeColor + ";font-weight:bold;font-size:0.8em;'>[" + escapeHtml(c.type) + "]</span> ";
                html += "<span style='font-size:0.85em;'>" + escapeHtml(c.axis || "") + "</span>";
                html += "</div>";
                html += "<div style='font-size:0.8em;color:var(--text-dim);'>" + escapeHtml(c.description || "") + "</div>";
                html += "</div>";
            });

            if (data.pending.validation) {
                var v = data.pending.validation;
                html += "<div style='margin-top:8px;font-size:0.8em;color:var(--text-dim);'>";
                html += "検証: 精度 " + ((v.accuracy_vs_current * 100).toFixed(1)) + "% / ";
                html += "<span style='color:#2ecc71;'>改善 " + (v.improved_count || 0) + "</span> / ";
                html += "<span style='color:#e74c3c;'>劣化 " + (v.degraded_count || 0) + "</span>";
                html += "</div>";
            }

            html += "<div style='margin-top:10px;display:flex;gap:6px;'>";
            html += "<button class='btn-sm btn-primary' id='drawer-refiner-approve'>全て承認</button>";
            html += "<button class='btn-sm' id='drawer-refiner-reject' style='background:#e74c3c;color:#fff;'>却下</button>";
            html += "</div>";

            drawerRefinerPending.innerHTML = html;

            var approveBtn = document.getElementById("drawer-refiner-approve");
            if (approveBtn) {
                approveBtn.addEventListener("click", function () {
                    if (!confirm("全ての変更を適用してタクソノミーを更新しますか？")) return;
                    fetch("/api/taxonomy/refine/approve", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
                        .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
                        .then(function (res) {
                            alert("承認: " + (res.accepted || 0) + " 件適用。タクソノミーが更新されました。");
                            loadDrawerRefiner();
                            checkSettingsAttention();
                        }).catch(function (err) { alert("Error: " + err.message); });
                });
            }

            var rejectBtn = document.getElementById("drawer-refiner-reject");
            if (rejectBtn) {
                rejectBtn.addEventListener("click", function () {
                    if (!confirm("提案を却下しますか？")) return;
                    fetch("/api/taxonomy/refine/reject", { method: "POST" })
                        .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
                        .then(function () {
                            alert("却下しました。");
                            loadDrawerRefiner();
                            checkSettingsAttention();
                        }).catch(function (err) { alert("Error: " + err.message); });
                });
            }
        }).catch(function () {});
    }

    if (reviewDrawerBtn) {
        reviewDrawerBtn.addEventListener("click", function () { toggleReviewDrawer(); });
    }
    if (drawerCloseBtn) {
        drawerCloseBtn.addEventListener("click", function () { toggleReviewDrawer(false); });
    }

    // Drawer tabs
    var drawerTabs = document.querySelectorAll(".drawer-tab");
    for (var dti = 0; dti < drawerTabs.length; dti++) {
        drawerTabs[dti].addEventListener("click", function () {
            var target = this.getAttribute("data-drawer-tab");
            drawerTabs.forEach(function (t) { t.classList.remove("active"); });
            document.querySelectorAll(".drawer-tab-content").forEach(function (c) { c.classList.remove("active"); });
            this.classList.add("active");
            var el = document.getElementById(target);
            if (el) el.classList.add("active");
        });
    }

    function loadNextReviewItem() {
        reviewState.editing = false;
        reviewContent.classList.remove("fade-in");
        reviewContent.classList.add("fade-out");

        setTimeout(function () {
            fetch("/api/review/queue?limit=1&offset=" + reviewState.skippedOffset)
                .then(function (r) {
                    if (!r.ok) throw new Error("HTTP " + r.status);
                    return r.json();
                })
                .then(function (data) {
                    var items = data.items || [];
                    if (items.length === 0) {
                        showReviewComplete();
                        return;
                    }
                    reviewState.currentItem = items[0];
                    renderReviewItem(items[0]);
                    updateReviewProgress();
                    reviewContent.classList.remove("fade-out");
                    reviewContent.classList.add("fade-in");
                })
                .catch(function (err) {
                    reviewContent.innerHTML = '<div class="review-complete"><h2 style="color:#e74c3c;">Error</h2>'
                        + '<p>' + escapeHtml(err.message) + '</p>'
                        + '<button class="btn-primary" onclick="this.closest(\'.review-content\').querySelector(\'div\').remove(); document.getElementById(\'review-close-btn\').click();">Close</button></div>';
                    reviewContent.classList.remove("fade-out");
                    reviewContent.classList.add("fade-in");
                });
        }, 150);
    }

    function renderReviewItem(item) {
        var s = item.session;
        var t = item.transcript || {};
        var segments = t.segments || [];
        var tags = (t.session || {}).tags || {};

        var priorityScore = s.priority_score || 0;
        var priorityClass = priorityScore >= 0.6 ? "high" : (priorityScore >= 0.35 ? "medium" : "low");
        var priorityLabel = priorityScore >= 0.6 ? "要確認" : (priorityScore >= 0.35 ? "確認推奨" : "低優先");

        var html = '<div class="review-session-header">';
        html += '<h2>' + escapeHtml(s.date) + ' ' + escapeHtml(s.time.replace(/-/g, ":"));
        html += ' <span class="review-priority-badge ' + priorityClass + '">' + priorityLabel + '</span>';
        html += '</h2>';
        html += '<div class="review-session-meta">';
        html += '<span>Duration: ' + formatDuration(s.duration_seconds) + '</span>';
        html += '<span>' + (s.segment_count || 0) + ' segments</span>';
        if (s.tags_quality) html += '<span>Quality: ' + escapeHtml(s.tags_quality) + '</span>';
        html += '</div></div>';

        // Current tags
        html += '<div class="review-current-tags">';
        html += '<div class="review-current-tags-label">Current Tags</div>';
        html += '<div class="detail-tags">';
        if (s.tags_conversation_type) html += '<span class="tag-badge conv-type">' + escapeHtml(s.tags_conversation_type) + '</span>';
        if (s.tags_outcomes) {
            s.tags_outcomes.split(",").forEach(function (o) {
                html += '<span class="tag-badge outcome">' + escapeHtml(o.trim()) + '</span>';
            });
        }
        if (s.tags_domains) {
            s.tags_domains.split(",").forEach(function (d) {
                html += '<span class="tag-badge domain">' + escapeHtml(d.trim()) + '</span>';
            });
        }
        if (s.tags_importance_v2) {
            var impClass = s.tags_importance_v2 === "高" ? "imp-high" : (s.tags_importance_v2 === "中" ? "imp-med" : "imp-low");
            html += '<span class="tag-badge ' + impClass + '">' + escapeHtml(s.tags_importance_v2) + '</span>';
        }
        html += '</div>';

        // Source info
        var sourceInfo = '';
        if (s.tags_tag_source) {
            sourceInfo += 'Source: ' + escapeHtml(s.tags_tag_source);
            if (s.tags_tag_source === 'rag' && s.tags_rag_similarity != null) {
                sourceInfo += ' (similarity: ' + Math.round(s.tags_rag_similarity * 100) + '%)';
            }
        }
        if (s.tags_quality) sourceInfo += (sourceInfo ? '  |  ' : '') + 'Quality: ' + escapeHtml(s.tags_quality);
        if (sourceInfo) html += '<div class="review-tag-source">' + sourceInfo + '</div>';
        html += '</div>';

        // Summary
        if (s.tags_summary) {
            html += '<div class="detail-summary">' + escapeHtml(s.tags_summary) + '</div>';
        }

        // Tag editor container (hidden by default, shown on E)
        html += '<div id="review-tag-editor-container" style="display:none;"></div>';

        // Transcript
        html += '<div class="review-transcript">';
        segments.forEach(function (seg) {
            var timeStr = formatTimestamp(seg.start || 0);
            var speaker = seg.person_name || "";
            var speakerClass = "unknown-speaker";
            if (speaker && speaker !== "unknown") {
                speakerClass = seg.is_user ? "user-speaker" : "known-speaker";
            }
            html += '<div class="segment">';
            html += '<span class="segment-time">' + timeStr + '</span>';
            if (speaker) {
                html += '<span class="speaker-badge ' + speakerClass + '">' + escapeHtml(speaker === "unknown" ? "?" : speaker) + '</span>';
            }
            html += '<span class="segment-text">' + escapeHtml(seg.text || "") + '</span>';
            html += '</div>';
        });
        html += '</div>';

        reviewContent.innerHTML = html;
    }

    function updateReviewProgress() {
        var total = reviewState.totalPending + reviewState.reviewedCount;
        if (total > 0) {
            var pct = Math.round((reviewState.reviewedCount / total) * 100);
            reviewProgressFill.style.width = pct + "%";
        } else {
            reviewProgressFill.style.width = "0%";
        }
        reviewProgressText.textContent = reviewState.reviewedCount + " / " + (reviewState.totalPending + reviewState.reviewedCount);
        reviewStatsText.textContent = "済: " + reviewState.reviewedCount + "  残: " + reviewState.totalPending;
    }

    function reviewAccept() {
        var item = reviewState.currentItem;
        if (!item) return;
        var s = item.session;

        reviewAcceptBtn.disabled = true;
        fetch("/api/review/accept", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({date: s.date, time: s.time}),
        }).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function () {
            reviewState.reviewedCount++;
            reviewState.totalPending = Math.max(0, reviewState.totalPending - 1);
            // Skipped items shift down since one was removed from the queue
            if (reviewState.skippedOffset > 0) reviewState.skippedOffset--;
            reviewAcceptBtn.disabled = false;
            loadNextReviewItem();
        }).catch(function (err) {
            showNotification("Error: " + err.message);
            reviewAcceptBtn.disabled = false;
        });
    }

    function reviewEdit() {
        var item = reviewState.currentItem;
        if (!item) return;
        var s = item.session;
        reviewState.editing = true;

        var container = document.getElementById("review-tag-editor-container");
        if (!container) return;
        if (container.style.display !== "none") {
            container.style.display = "none";
            reviewState.editing = false;
            return;
        }
        if (!taxonomyCache) {
            container.innerHTML = '<p class="text-dim">Taxonomy not loaded</p>';
            container.style.display = "block";
            return;
        }

        var currentTags = {
            conversation_type: s.tags_conversation_type,
            outcomes: s.tags_outcomes ? s.tags_outcomes.split(",") : [],
            domains: s.tags_domains ? s.tags_domains.split(",") : [],
            importance: s.tags_importance_v2,
        };

        var html = '<div class="review-tag-editor">';
        html += '<div class="tag-editor-row"><label>会話の種類</label><select id="rte-conv-type">';
        (taxonomyCache.conversation_type || []).forEach(function (v) {
            html += '<option value="' + escapeHtml(v) + '"' + (currentTags.conversation_type === v ? ' selected' : '') + '>' + escapeHtml(v) + '</option>';
        });
        html += '</select></div>';
        html += '<div class="tag-editor-row"><label>成果物</label><div class="te-checks">';
        (taxonomyCache.outcomes || []).forEach(function (v) {
            var checked = currentTags.outcomes.indexOf(v) !== -1 ? ' checked' : '';
            html += '<label class="te-check"><input type="checkbox" value="' + escapeHtml(v) + '"' + checked + ' class="rte-outcome-cb"> ' + escapeHtml(v) + '</label>';
        });
        html += '</div></div>';
        html += '<div class="tag-editor-row"><label>事業領域</label><div class="te-checks">';
        (taxonomyCache.domains || []).forEach(function (v) {
            var checked = currentTags.domains.indexOf(v) !== -1 ? ' checked' : '';
            html += '<label class="te-check"><input type="checkbox" value="' + escapeHtml(v) + '"' + checked + ' class="rte-domain-cb"> ' + escapeHtml(v) + '</label>';
        });
        html += '</div></div>';
        html += '<div class="tag-editor-row"><label>重要度</label><select id="rte-importance">';
        (taxonomyCache.importance || []).forEach(function (v) {
            html += '<option value="' + escapeHtml(v) + '"' + (currentTags.importance === v ? ' selected' : '') + '>' + escapeHtml(v) + '</option>';
        });
        html += '</select></div>';
        html += '<div class="tag-editor-actions">';
        html += '<button class="btn-primary" id="rte-save">保存して承認</button>';
        html += ' <button class="btn-sm" id="rte-cancel">キャンセル</button>';
        html += ' <span id="rte-status"></span>';
        html += '</div></div>';

        container.innerHTML = html;
        container.style.display = "block";

        document.getElementById("rte-cancel").addEventListener("click", function () {
            container.style.display = "none";
            reviewState.editing = false;
        });

        document.getElementById("rte-save").addEventListener("click", function () {
            var convType = document.getElementById("rte-conv-type").value;
            var importance = document.getElementById("rte-importance").value;
            var outcomes = [];
            var outCbs = container.querySelectorAll(".rte-outcome-cb:checked");
            for (var i = 0; i < outCbs.length; i++) outcomes.push(outCbs[i].value);
            var domains = [];
            var domCbs = container.querySelectorAll(".rte-domain-cb:checked");
            for (var j = 0; j < domCbs.length; j++) domains.push(domCbs[j].value);

            var statusEl = document.getElementById("rte-status");
            statusEl.textContent = "Saving...";

            // First update tags, then accept
            fetch("/api/sessions/" + encodeURIComponent(s.date) + "/" + encodeURIComponent(s.time) + "/session-tags", {
                method: "PUT",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({conversation_type: convType, outcomes: outcomes, domains: domains, importance: importance}),
            }).then(function (r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            }).then(function (data) {
                if (data.error) throw new Error(data.error);
                // Tags saved — session is now marked as manually_corrected via PUT handler
                reviewState.reviewedCount++;
                reviewState.totalPending = Math.max(0, reviewState.totalPending - 1);
                if (reviewState.skippedOffset > 0) reviewState.skippedOffset--;
                container.style.display = "none";
                reviewState.editing = false;
                loadNextReviewItem();
            }).catch(function (err) {
                statusEl.textContent = "Error: " + err.message;
                statusEl.style.color = "#e74c3c";
            });
        });
    }

    function reviewSkip() {
        reviewState.skippedOffset++;
        fetch("/api/review/skip", { method: "POST" }).catch(function () {});
        loadNextReviewItem();
    }

    function showReviewComplete() {
        reviewContent.classList.remove("fade-out");
        reviewContent.classList.add("fade-in");
        reviewContent.innerHTML = '<div class="review-complete">'
            + '<h2>All Done!</h2>'
            + '<div class="review-complete-stats">'
            + '<p>今回のレビュー: ' + reviewState.reviewedCount + ' 件承認</p>'
            + '<p>スキップ: ' + reviewState.skippedOffset + ' 件 (次回再表示されます)</p>'
            + '</div>'
            + '<button class="btn-primary" id="review-complete-close">閉じる</button>'
            + '</div>';
        var closeBtn = document.getElementById("review-complete-close");
        if (closeBtn) closeBtn.addEventListener("click", closeReviewMode);
        updateReviewProgress();
    }

    // Event listeners
    if (reviewBtn) reviewBtn.addEventListener("click", openReviewMode);
    if (reviewCloseBtn) reviewCloseBtn.addEventListener("click", closeReviewMode);
    if (reviewAcceptBtn) reviewAcceptBtn.addEventListener("click", reviewAccept);
    if (reviewEditBtn) reviewEditBtn.addEventListener("click", reviewEdit);
    if (reviewSkipBtn) reviewSkipBtn.addEventListener("click", reviewSkip);

    // ============================================================
    // Notification helper
    // ============================================================

    function showNotification(msg) {
        var existing = document.querySelector(".notification");
        if (existing) existing.remove();
        var el = document.createElement("div");
        el.className = "notification";
        el.textContent = msg;
        document.body.appendChild(el);
        setTimeout(function () { el.classList.add("show"); }, 10);
        setTimeout(function () {
            el.classList.remove("show");
            setTimeout(function () { el.remove(); }, 300);
        }, 3000);
    }

    // --- Settings attention badge ---

    var settingsAttentionBadge = document.getElementById("settings-attention-badge");
    var drawerAttentionBadge = document.getElementById("drawer-attention-badge");
    var navDotSuggestions = document.getElementById("nav-dot-suggestions");
    var navDotRag = document.getElementById("nav-dot-rag");
    var navDotRefiner = document.getElementById("nav-dot-refiner");
    var navDotVoice = document.getElementById("nav-dot-voice");

    function checkSettingsAttention() {
        var count = 0;
        var done = 0;
        var total = 4;

        function update() {
            done++;
            if (done < total) return;
            // Update both header badge and drawer button badge
            [settingsAttentionBadge, drawerAttentionBadge].forEach(function (badge) {
                if (!badge) return;
                if (count > 0) {
                    badge.textContent = count;
                    badge.classList.remove("hidden");
                } else {
                    badge.classList.add("hidden");
                }
            });
        }

        // 1. Voice: 0 speakers registered
        fetch("/api/speakers/status").then(function (r) {
            if (!r.ok) return null;
            return r.json();
        }).then(function (data) {
            if (!data || data.speaker_count === 0) {
                count++;
                if (navDotVoice) navDotVoice.classList.remove("hidden");
            } else {
                if (navDotVoice) navDotVoice.classList.add("hidden");
            }
            update();
        }).catch(function () { update(); });

        // 2. LLM Suggestions pending
        fetch("/api/suggestions").then(function (r) { return r.json(); }).then(function (data) {
            if (data && data.length > 0) {
                count++;
                if (navDotSuggestions) navDotSuggestions.classList.remove("hidden");
            } else {
                if (navDotSuggestions) navDotSuggestions.classList.add("hidden");
            }
            update();
        }).catch(function () { update(); });

        // 3. RAG not built
        fetch("/api/rag/status").then(function (r) {
            if (!r.ok) return null;
            return r.json();
        }).then(function (data) {
            if (!data || !data.ready) {
                count++;
                if (navDotRag) navDotRag.classList.remove("hidden");
            } else {
                if (navDotRag) navDotRag.classList.add("hidden");
            }
            update();
        }).catch(function () { update(); });

        // 4. Refiner has pending proposals
        fetch("/api/taxonomy/refine/status").then(function (r) {
            if (!r.ok) return null;
            return r.json();
        }).then(function (data) {
            if (data && data.has_pending) {
                count++;
                if (navDotRefiner) navDotRefiner.classList.remove("hidden");
            } else {
                if (navDotRefiner) navDotRefiner.classList.add("hidden");
            }
            update();
        }).catch(function () { update(); });
    }

    // --- Init ---
    Promise.all([loadSpeakerNames(), loadTaxonomy(), loadAudioSceneLabels()]).then(function () {
        loadPage(1);
    });
    checkBatchStatus("diarize");
    checkBatchStatus("tags");
    loadHotwordsStatus();
    loadReviewStats();
    checkSettingsAttention();
})();
