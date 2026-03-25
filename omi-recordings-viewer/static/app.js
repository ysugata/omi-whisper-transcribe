(function () {
    "use strict";

    const listEl = document.getElementById("session-list");
    const paginationEl = document.getElementById("pagination");
    const searchInput = document.getElementById("search-input");
    const searchBtn = document.getElementById("search-btn");
    const modal = document.getElementById("detail-modal");
    const modalClose = document.getElementById("modal-close");
    const detailHeader = document.getElementById("detail-header");
    const audioContainer = document.getElementById("audio-player-container");
    const segmentsContainer = document.getElementById("segments-container");

    let currentPage = 1;
    let currentQuery = "";
    let audioEl = null;

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
        // date: "2026-03-25", time: "09-21-29"
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

        // Attach click handlers
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

        // Show page numbers with ellipsis
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

        audioContainer.innerHTML =
            '<audio id="audio-el" controls preload="metadata" src="/audio/' + s.date + "/" + s.time + '/audio.wav"></audio>';
        audioEl = document.getElementById("audio-el");

        // Track active segment
        audioEl.addEventListener("timeupdate", onTimeUpdate);

        // Render segments
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

        // Timestamp click → seek
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
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") closeModal();
    });

    // --- Init ---
    loadPage(1);
})();
