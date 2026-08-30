#!/usr/bin/env python3
"""Watch the docs folder and push changes to Telegram (topic 307) directly.

This script does TWO jobs so the cron agent does not need to reason about
sending:
  1. Diff /chista/Chista/ChistaAgent/docs against the stored snapshot.
  2. For each change, send the file to Telegram via Bot API with caption =
     the relative path (docs/<rel>). On UPDATE, delete the previous message
     first (message id read from the registry). On DELETE, do NOT delete on
     Telegram -- just report the deletion time in stdout.
  3. Update the snapshot + registry so the next run is stable.

Token is read from /tmp/.tg_token (first non-empty line).
Registry: /home/amirelite_ai/.hermes/.telegram_sent_files.json
Snapshot: /chista/Chista/ChistaAgent/.docs_watch_state.json

Stdout is a human-readable report (Persian) for the cron delivery.
"""
import os, hashlib, json, time, datetime, urllib.request, urllib.parse

ROOT = "/chista/Chista/ChistaAgent/docs"
SNAP = "/chista/Chista/ChistaAgent/.docs_watch_state.json"
REG = "/home/amirelite_ai/.hermes/.telegram_sent_files.json"
TOKEN_FILE = "/tmp/.tg_token"
CHAT = "-1003924862595"
THREAD = 307


def load_token():
    with open(TOKEN_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    raise RuntimeError("token not found")


TOKEN = load_token()


def tg(method, **params):
    url = f"https://api.telegram.org/bot{TOKEN}/{method}"
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(url, data=data)
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:200]}


def send_doc(path, caption):
    boundary = "----hermesdocswatch"
    def mp(field, filename, data, ctype="text/plain"):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    body = mp("document", os.path.basename(path), open(path, "rb").read(), "text/plain; charset=utf-8")
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{CHAT}\r\n".encode()
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"message_thread_id\"\r\n\r\n{THREAD}\r\n".encode()
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode()
    body += f"--{boundary}--\r\n".encode()
    url = f"https://api.telegram.org/bot{TOKEN}/sendDocument"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def scan():
    cur = {}
    for dp, _, fns in os.walk(ROOT):
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, ROOT)
            try:
                data = open(p, "rb").read()
                cur[rel] = {"full": p, "size": len(data), "mtime": os.path.getmtime(p),
                            "sha": hashlib.sha256(data).hexdigest()}
            except Exception:
                pass
    return cur


def iso(t):
    return datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S UTC") if t else "?"


def main():
    cur = scan()
    try:
        snap = json.load(open(SNAP))
    except Exception:
        snap = {"files": {}, "deleted": {}}
    prev = snap.get("files", {})
    deleted = snap.get("deleted", {})
    try:
        reg = json.load(open(REG))
    except Exception:
        reg = {}

    new, updated, gone = [], [], []
    for rel, info in cur.items():
        if rel not in prev:
            new.append(rel)
        elif prev[rel].get("sha") != info["sha"]:
            updated.append(rel)
    for rel in prev:
        if rel not in cur:
            gone.append(rel)

    now = time.time()
    report = []
    for rel in new:
        cap = "docs/" + rel
        res = send_doc(cur[rel]["full"], cap)
        if res.get("ok"):
            mid = res["result"]["message_id"]
            reg[cap] = {"message_id": mid, "chat_id": CHAT, "thread_id": THREAD, "caption": cap, "path": cur[rel]["full"]}
            report.append(f"📄 New file sent: {cap}")
        else:
            report.append(f"⚠️ Send error {cap}: {res.get('description')}")
    for rel in updated:
        cap = "docs/" + rel
        old_mid = reg.get(cap, {}).get("message_id")
        if old_mid:
            tg("deleteMessage", chat_id=CHAT, message_id=old_mid)
        res = send_doc(cur[rel]["full"], cap)
        if res.get("ok"):
            mid = res["result"]["message_id"]
            reg[cap] = {"message_id": mid, "chat_id": CHAT, "thread_id": THREAD, "caption": cap, "path": cur[rel]["full"]}
            report.append(f"🔄 Updated: {cap}")
        else:
            report.append(f"⚠️ Update error {cap}: {res.get('description')}")
    for rel in gone:
        cap = "docs/" + rel
        deleted[rel] = {"deleted_at": now, "last_seen": prev[rel].get("mtime"), "size": prev[rel].get("size")}
        # remove from registry entirely (file no longer exists; we do NOT delete the Telegram message)
        reg.pop(cap, None)
        report.append(f"🗑️ File deleted (not deleted in Telegram): {cap} — deleted at: {iso(now)}")

    new_files = {rel: {"size": cur[rel]["size"], "mtime": cur[rel]["mtime"], "sha": cur[rel]["sha"]} for rel in cur}
    json.dump({"files": new_files, "deleted": deleted, "updated_at": now}, open(SNAP, "w"), indent=2)
    json.dump(reg, open(REG, "w"), ensure_ascii=False, indent=2)

    if report:
        print("\n".join(report))
    else:
        print("NOCHANGE")


if __name__ == "__main__":
    main()
