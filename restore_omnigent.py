import hashlib, json, os, sys

with open('omnigent_current.zip.manifest.json') as f:
    m = json.load(f)

out = m['filename']
sha = hashlib.sha256()

with open(out, 'wb') as dst:
    for part in m['parts']:
        with open(part, 'rb') as src:
            data = src.read()
            sha.update(data)
            dst.write(data)

digest = sha.hexdigest()
if digest != m['sha256']:
    print(f'SHA-256 MISMATCH: expected {m["sha256"]}, got {digest}')
    sys.exit(1)

print(f'Restored {out} ({os.path.getsize(out)} bytes) — SHA-256 OK')
