import os, json, re, uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


load_dotenv("feit.env")
from openai import OpenAI
client = OpenAI(api_key=os.environ["AI_API_KEY"],
                base_url="https://openrouter.ai/api/v1")
MODEL = os.getenv("AI_MODEL", "google/gemini-2.5-flash")

app = FastAPI()
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def show_errors(request, exc):
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
S = {}
CFG = dict(low=0.4, high=0.6, conf=0.6, hint=0.5, max_turns=6, gap=0.3)  # starting values, not validated

LESSON = ("MITRE ATT&CK describes how attackers operate. A TACTIC is the attacker's goal at a stage (e.g. gain initial access); "
          "a TECHNIQUE is how they do it (e.g. phishing). Attacks typically move from getting in, to running code, staying in, "
          "stealing credentials, moving sideways, communicating with outside servers, and taking data, though not in a rigid order. "
          "Every stage is a chance for defenders to detect or disrupt the attack.")

RUBRIC = {
 "i1": dict(name="Tactic vs technique", q="Is phishing a tactic or a technique?",
   crit="A tactic is the attacker's goal at a stage; a technique is the specific way they achieve it.",
   fix="A tactic is the WHY (goal, e.g. initial access). A technique is the HOW (e.g. phishing)."),
 "i2": dict(name="Typical, not rigid order", q="Do attackers always go through the stages in the same order?",
   crit="Stages follow a typical progression (get in, execute, persist, steal credentials, move sideways, communicate, exfiltrate) but not a rigid order.",
   fix="The order is a common pattern, not a checklist. Attackers can skip, repeat or loop back."),
 "i3": dict(name="Every stage is a chance to stop them", q="If they're already inside, isn't it too late?",
   crit="Each stage is an opportunity to detect or disrupt; catching an earlier stage costs the attacker more.",
   fix="Defenders can detect or break the chain at any stage, and earlier is cheaper than later."),
 "i4": dict(name="Stolen credentials enable lateral movement", q="Why would they steal passwords if they're already in?",
   crit="Stolen credentials let attackers move sideways as a legitimate user without needing new exploits.",
   fix="With valid logins, attackers move between systems looking like normal users, which is hard to spot."),
 "i5": dict(name="Persistence vs privilege escalation", q="If I restart the computer, what happens to the attacker?",
   crit="Persistence keeps access across restarts or resets; privilege escalation raises permissions. They are different.",
   fix="Persistence = keeping access after restarts. Privilege escalation = gaining higher permissions."),
 "i6": dict(name="Name the tactic from an event", q="Someone uploads a huge archive to an outside server. What stage is that?",
   crit="Can identify the tactic from a described event by its goal, not the tool used (e.g. large upload to outside server = exfiltration).",
   fix="Classify by the goal. Copying data out to an outside server is exfiltration, whatever tool was used."),
}

def llm(system, user, temp=0.2, tokens=600):
    r = client.chat.completions.create(
        model=MODEL, max_tokens=max(tokens, 1500),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content.strip()

def grade(text):
    rub = "\n".join(f"{k}: {v['crit']}" for k, v in RUBRIC.items())
    sys = ("You grade a learner's explanation of MITRE ATT&CK against a rubric. Credit an idea only if the LEARNER clearly states it "
           "in their own words. For each idea give status (correct|partial|missing|wrong) and evidence: an exact verbatim quote from "
           'the learner text, or null. Return ONLY JSON like {"i1":{"status":"missing","evidence":null},...}.')
    try:
        g = json.loads(re.search(r"\{.*\}", llm(sys, f"RUBRIC:\n{rub}\n\nLEARNER TEXT:\n{text}", 0, 900), re.S).group())
    except (json.JSONDecodeError, AttributeError):
        g = {}
    out = {}
    for k in RUBRIC:
        d = g.get(k) or {}
        st, ev = d.get("status", "missing"), d.get("evidence")
        if st in ("correct", "partial") and (not ev or ev.lower() not in text.lower()):
            st = "missing"  # no verbatim evidence, no credit
        out[k] = {"status": st if st in ("correct", "partial", "wrong") else "missing", "evidence": ev}
    return out

def score(g):
    return sum({"correct": 1, "partial": .5}.get(v["status"], 0) for v in g.values()) / len(g)

def weakest(g, skip=()):
    order = {"wrong": 0, "missing": 1, "partial": 2, "correct": 3}
    cand = [i for i in RUBRIC if i not in skip and g[i]["status"] != "correct"]
    return min(cand, key=lambda i: order[g[i]["status"]]) if cand else None

class Start(BaseModel):
    confidence: int

class Msg(BaseModel):
    sid: str
    message: str = ""

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/api/content")
def content():
    return {"lesson": LESSON, "ideas": {k: v["name"] for k, v in RUBRIC.items()}}

@app.post("/api/start")
def start(b: Start):
    sid = uuid.uuid4().hex[:8]
    S[sid] = dict(c=b.confidence / 5, teach=[], refl=[], turns=0, g1=None, weak=None, asked=[])
    return {"sid": sid}

@app.post("/api/explain")
def explain(b: Msg):
    s = S[b.sid]; s["teach"].append(b.message); s["turns"] += 1
    g = s["g1"] = grade("\n".join(s["teach"]))
    t = weakest(g)
    t = weakest(g, s["asked"])
    if t is None or s["turns"] >= CFG["max_turns"]:
        return {"grade": g, "reply": "Thanks, that's all my questions for now.", "done": True}
    s["asked"].append(t)
    sys = ("You are a dumb curious newcomer with no security knowledge. In at most 2 short sentences: briefly acknowledge what the learner "
           "just said, then ask exactly ONE follow-up question. Only use words the learner has already used. Never state or hint at the "
           f"answer. Ask about this: {RUBRIC[t]['q']}")
    return {"grade": g, "reply": llm(sys, "Conversation so far (learner messages):\n" + "\n".join(s["teach"]), 0.6, 200), "done": False}

@app.post("/api/feedback")
def feedback(b: Msg):
    s = S[b.sid]; g = s["g1"] or grade("\n".join(s["teach"]))
    s["g1"], s["weak"] = g, weakest(g)
    sys = ('You are a supportive expert coach. Return ONLY JSON: {"feedback": "...", "question": "..."}. '
           "feedback: 2 sentences max; say what the learner explained well and the single biggest gap. Do NOT give the answer. "
           'question: ONE reflective question about the biggest gap, or "" if there is no meaningful gap.')
    info = json.dumps({k: {"idea": RUBRIC[k]["crit"], **v} for k, v in g.items()})
    raw = llm(sys, f"Grader results:\n{info}\nBiggest gap: {s['weak']}")
    try:
        d = json.loads(re.search(r"\{.*\}", raw, re.S).group())
    except Exception:
        d = {"feedback": raw, "question": ""}
    return {"feedback": d.get("feedback", ""), "question": d.get("question", "") if s["weak"] else ""}

@app.post("/api/reflect")
def reflect(b: Msg):
    s = S[b.sid]; s["refl"].append(b.message)
    g1 = s["g1"]; g2 = grade("\n".join(s["teach"] + s["refl"]))
    k1, k2 = score(g1), score(g2)
    k = k1 + CFG["hint"] * (k2 - k1)  # hint-assisted ideas count half for routing
    c, w = s["c"], s["weak"]
    if k < CFG["low"]: route = "lesson_full" if c < CFG["conf"] else "lesson_targeted_reveal"
    elif k < CFG["high"]: route = "lesson_targeted"
    else: route = "sim_guided" if c < CFG["conf"] else "sim_unguided"
    missed = [{"name": RUBRIC[i]["name"], "fix": RUBRIC[i]["fix"]} for i in RUBRIC if g2[i]["status"] != "correct"]
    prompted = [RUBRIC[i]["name"] for i in RUBRIC if score({i: g2[i]}) > score({i: g1[i]})]
    coach = ""
    if w:
        sys = ("You are a supportive expert coach. The learner replied to your reflective question. If they now show the idea, confirm it "
               "and add one nuance. If they don't (or say they don't know), explain it plainly using the snippet, then ask them to restate it in one sentence. Max 4 sentences.")
        coach = llm(sys, f"Idea: {RUBRIC[w]['crit']}\nSnippet: {RUBRIC[w]['fix']}\nLearner reply: {b.message}", 0.4, 350)
    return {"reply": coach, "grade": g2, "route": route, "k1": k1, "k2": k2, "k": k, "c": c,
            "gap": round(c - k, 2), "miscalibrated": abs(c - k) > CFG["gap"], "missed": missed, "prompted": prompted}
