// Cloudflare Worker for fuebc-translate.org.
// Passes every request through to the church Mac (via the Cloudflare Tunnel).
// When the Mac is off or server.py isn't running, Cloudflare gets an error back
// (530 = tunnel offline, 502/503/504 = server not answering) and this shows a
// friendly bilingual "offline" page instead of Cloudflare's error screen.

const OFFLINE_STATUSES = [502, 503, 504, 530];

export default {
    async fetch(request) {
        const isWebSocket = request.headers.get("Upgrade") === "websocket";

        let response;
        try {
            response = await fetch(request);
        } catch (err) {
            if (isWebSocket) return new Response("Translation server offline", { status: 503 });
            return offlinePage();
        }

        // Live audio/transcript sockets just retry on their own, so pass them through untouched
        if (isWebSocket || !OFFLINE_STATUSES.includes(response.status)) return response;
        return offlinePage();
    },
};

function offlinePage() {
    return new Response(OFFLINE_HTML, {
        status: 503,
        headers: {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "no-store",
            "Retry-After": "60",
        },
    });
}

const OFFLINE_HTML = `<!doctype html>
<html lang="uk">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="60">
    <title>Live Church Translation</title>
    <style>
        :root {
            --ink: 20, 22, 26;
            --bg: #f4f3ef;
            --card: #ffffff;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --ink: 236, 237, 240;
                --bg: #16181c;
                --card: #1f2227;
            }
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px 16px;
            background: var(--bg);
            color: rgb(var(--ink));
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .card {
            width: 100%;
            max-width: 440px;
            background: var(--card);
            border: 1px solid rgba(var(--ink), .1);
            border-radius: 18px;
            padding: 32px 28px;
        }
        .status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            letter-spacing: .12em;
            text-transform: uppercase;
            color: rgba(var(--ink), .55);
        }
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 99px;
            background: rgba(var(--ink), .3);
        }
        h1 {
            margin: 18px 0 8px;
            font-size: 24px;
            font-weight: 500;
            line-height: 1.25;
        }
        p {
            margin: 0;
            font-size: 15px;
            line-height: 1.55;
            color: rgba(var(--ink), .7);
        }
        hr {
            border: 0;
            border-top: 1px solid rgba(var(--ink), .1);
            margin: 24px 0;
        }
        .note {
            margin-top: 24px;
            font-size: 13px;
            color: rgba(var(--ink), .5);
        }
    </style>
</head>
<body>
    <main class="card">
        <div class="status"><span class="dot"></span>Offline · Не в ефірі</div>

        <h1>Переклад зараз не транслюється</h1>
        <p>Живий переклад доступний під час богослужінь. Будь ласка, поверніться, коли служіння почнеться.</p>

        <hr>

        <h1 lang="en">Translation isn't live right now</h1>
        <p lang="en">Live translation is available during church services. Please come back when the service begins.</p>

        <p class="note">This page checks again every minute and opens the translation automatically once it's live. · Сторінка оновлюється щохвилини.</p>
    </main>
</body>
</html>`;
