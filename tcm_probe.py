# -*- coding: utf-8 -*-
"""
真接入探针:注册->登录->建讯飞embedding provider->建数据集->传文档->解析->检索"桂枝汤"。
全部通过 RAGFlow 自己暴露的 REST API 完成,不改 RAGFlow 一行源码。
"""
import os
import sys
import time
import json
import base64
import requests

BASE = "http://localhost:9380/api/v1"


def rsa_encrypt(password: str, pub_pem_path: str) -> str:
    from Cryptodome.PublicKey import RSA
    from Cryptodome.Cipher import PKCS1_v1_5
    rsa_key = RSA.importKey(open(pub_pem_path, "r", encoding="utf-8").read(), "Welcome")
    cipher = PKCS1_v1_5.new(rsa_key)
    b64_pwd = base64.b64encode(password.encode("utf-8")).decode("utf-8")
    enc = cipher.encrypt(b64_pwd.encode("utf-8"))
    return base64.b64encode(enc).decode("utf-8")


def step(name):
    print("\n=== STEP: %s ===" % name, flush=True)


def show(r, limit=800):
    print("HTTP", r.status_code, flush=True)
    try:
        print(json.dumps(r.json(), ensure_ascii=False)[:limit], flush=True)
    except Exception:
        print(r.text[:limit], flush=True)


def main():
    xf_key = os.environ["XF_EMBED_KEY"]
    xf_base = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
    xf_embed_model = "xop3qwen8bembedding"
    pub_pem = sys.argv[1]

    s = requests.Session()
    email = "probe_tcm@example.com"
    pwd_plain = "Probe#TCM2026_zw!"
    enc_pwd = rsa_encrypt(pwd_plain, pub_pem)

    step("register")
    r = s.post(BASE + "/users", json={"nickname": "tcm_probe", "email": email, "password": enc_pwd})
    show(r)

    step("login")
    r = s.post(BASE + "/auth/login", json={"email": email, "password": enc_pwd})
    show(r)

    step("new api token")
    r = s.post(BASE + "/system/tokens", params={"name": "probe"})
    show(r)
    tok = r.json()["data"]["token"]
    hdr = {"Authorization": "Bearer " + tok}

    step("add provider OpenAI-API-Compatible")
    r = s.put(BASE + "/providers", json={"provider_name": "OpenAI-API-Compatible"}, headers=hdr)
    show(r)

    step("create provider instance (xfyun embedding gateway) WITH model_info so verify_api_key has a model to test")
    r = s.post(BASE + "/providers/OpenAI-API-Compatible/instances", json={
        "instance_name": "xfyun",
        "api_key": xf_key,
        "base_url": xf_base,
        "model_info": [{
            "model_type": ["embedding"],
            "model_name": xf_embed_model,
            "max_tokens": 8192,
        }],
    }, headers=hdr)
    show(r)
    if r.json().get("code") != 0:
        print("!!! provider instance creation FAILED, aborting early with clear reason !!!", flush=True)
        sys.exit(1)

    step("set tenant default embedding model")
    r = s.patch(BASE + "/models/default", json={
        "model_provider": "OpenAI-API-Compatible",
        "model_instance": "xfyun",
        "model_name": xf_embed_model,
        "model_type": "embedding",
    }, headers=hdr)
    show(r)
    if r.json().get("code") != 0:
        print("!!! set default embedding model FAILED, aborting early with clear reason !!!", flush=True)
        sys.exit(1)

    step("create dataset")
    r = s.post(BASE + "/datasets", json={"name": "tcm_probe_ds", "chunk_method": "naive"}, headers=hdr)
    show(r)
    ds = r.json()["data"]["id"]

    step("upload document (original TCM formula notes, self-authored)")
    content = (
        "桂枝汤方义速览(本测试自撰摘要,非古籍原文引用):\n"
        "桂枝汤为张仲景《伤寒论》中调和营卫的基础方,由桂枝、芍药、炙甘草、生姜、大枣五味药组成,"
        "用于外感风寒、营卫不和引起的发热、汗出、恶风、脉浮缓等表现。\n"
        "麻黄汤方义速览:由麻黄、桂枝、炙甘草、杏仁组成,用于外感风寒表实、无汗而喘的情况,"
        "与桂枝汤为表虚、表实相对的一组方剂。\n"
        "大承气汤方义速览:由大黄、芒硝、枳实、厚朴组成,属于攻下法代表方,"
        "用于阳明腑实、燥屎内结的里实热证,与前两方的解表功能不同。\n"
    )
    files = {"file": ("tcm_notes.txt", content.encode("utf-8"), "text/plain")}
    r = s.post(BASE + "/datasets/%s/documents" % ds, files=files, headers=hdr)
    show(r)
    doc_id = r.json()["data"][0]["id"]

    step("trigger parse")
    r = s.post(BASE + "/datasets/%s/documents/parse" % ds, json={"document_ids": [doc_id]}, headers=hdr)
    show(r)

    step("poll parse status")
    done = False
    for i in range(30):
        time.sleep(6)
        r = s.get(BASE + "/datasets/%s/documents" % ds, headers=hdr)
        data = r.json().get("data", {})
        docs = data.get("docs", data) if isinstance(data, dict) else data
        d0 = docs[0] if docs else {}
        run = d0.get("run")
        progress = d0.get("progress")
        progress_msg = d0.get("progress_msg")
        print("poll %d: run=%s progress=%s msg=%s" % (i, run, progress, progress_msg), flush=True)
        print("poll %d full doc record: %s" % (i, json.dumps(d0, ensure_ascii=False)), flush=True)
        if str(run) in ("3", "DONE", "done"):
            done = True
            break
        if str(run) in ("4", "FAIL", "fail"):
            print("PARSE FAILED:", json.dumps(d0, ensure_ascii=False), flush=True)
            break
    print("parse done:", done, flush=True)

    step("retrieval search: 桂枝汤")
    r = s.post(BASE + "/datasets/%s/search" % ds, json={"question": "桂枝汤的组成和主治是什么？", "top_k": 5}, headers=hdr)
    show(r, limit=4000)

    print("\n=== PROBE FINISHED ===", flush=True)


if __name__ == "__main__":
    main()
