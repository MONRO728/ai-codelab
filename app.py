# -*- coding: utf-8 -*-
"""AI CodeLab
# 🧪 AI CodeLab
**Test your knowledge. Discover your potential.**

AI CodeLab — Google Gemini yordamida IT mavzularida test yaratish, testni bajarish va natijalarni tahlil qilish uchun interaktiv platforma.

### 🚀 Hugging Face Spaces'ga joylash
1. Space yaratayotganda SDK sifatida **Gradio** ni tanlang.
2. `app.py` va `requirements.txt` fayllarini Space'ga yuklang.
3. Space sozlamalaridagi **Variables and secrets** bo'limiga `GEMINI_API_KEY` nomli secret qo'shing.
"""

import os
from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable sozlanmagan.")
client = genai.Client(api_key=API_KEY)
print("✅ API kaliti topildi, Gemini klienti yaratildi.")

"""## ⚙️ 3. Konfiguratsiya
`MODEL` o'zgaruvchisi orqali modelni o'zgartirishingiz mumkin. Kod modelni API'dagi haqiqiy ro'yxat bilan tekshiradi; model mavjud bo'lmasa, mavjud Flash modellardan biriga o'tadi yoki aniq xatolik beradi.
"""

import json
import re
import random
import html
import traceback

# ---- Model sozlamasi ----
MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]

# ---- Test sozlamalari ----
TOPICS = [
    "Python", "JavaScript", "HTML/CSS", "SQL", "Kompyuter tarmoqlari",
    "Sun'iy intellekt", "Algoritmlar", "Kiberxavfsizlik",
]
OTHER_TOPIC = "Boshqa mavzu"
DIFFICULTIES = ["Boshlang'ich", "O'rta", "Murakkab"]
DIFFICULTY_HINTS = {
    "Boshlang'ich": "asosiy tushunchalar va oddiy misollar, yangi o'rganuvchilar uchun",
    "O'rta": "amaliy bilim, kod o'qish, tushunchalarni qo'llash",
    "Murakkab": "chuqur bilim, murakkab holatlar, nozik jihatlar va optimallashtirish",
}
QUESTION_COUNTS = [5, 10, 15, 20]
PRACTICE = "Practice Mode"
EXAM = "Exam Mode"
MODES = [PRACTICE, EXAM]
LETTERS = ["A", "B", "C", "D"]
MAX_ATTEMPTS = 3

# (minimal foiz, baho, rang)
GRADES = [
    (90, "A'lo", "#22c55e"),
    (75, "Juda yaxshi", "#06b6d4"),
    (60, "Yaxshi", "#6366f1"),
    (40, "Qoniqarli", "#f59e0b"),
    (0, "Ko'proq mashq qilish kerak", "#ef4444"),
]


def list_available_models():
    """API orqali generateContent'ni qo'llaydigan modellar ro'yxatini qaytaradi."""
    try:
        all_models = list(client.models.list())
    except Exception as exc:
        raise RuntimeError(
            "Modellar ro'yxatini olib bo'lmadi. API kalit to'g'riligini va internet "
            f"aloqasini tekshiring. (Tafsilot: {str(exc)[:200]})"
        ) from exc
    names_all, names_gen = [], []
    for m in all_models:
        name = (getattr(m, "name", "") or "").replace("models/", "")
        if not name:
            continue
        names_all.append(name)
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            names_gen.append(name)
    return names_gen or names_all


def resolve_model(preferred: str) -> str:
    """MODEL mavjud bo'lsa — shuni, bo'lmasa API'dagi mavjud Flash modelni tanlaydi."""
    available = list_available_models()
    if preferred in available:
        return preferred
    for cand in FALLBACK_MODELS:
        if cand in available:
            print(f"⚠️ '{preferred}' mavjud emas, '{cand}' tanlandi.")
            return cand
    flash = [n for n in available if re.fullmatch(r"gemini-[\d.]+-flash(-lite)?", n)]
    if flash:
        def version_key(n):
            return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", n)][:1] + [0 if "lite" in n else 1]
        flash.sort(key=version_key, reverse=True)
        print(f"⚠️ '{preferred}' mavjud emas, '{flash[0]}' tanlandi.")
        return flash[0]
    raise RuntimeError(
        f"❌ '{preferred}' modeli API'da topilmadi va mos zaxira model ham yo'q. "
        "MODEL o'zgaruvchisini quyidagi mavjud modellardan biriga o'zgartiring:\n  "
        + "\n  ".join(available[:30])
    )


ACTIVE_MODEL = resolve_model(MODEL)
print(f"✅ Ishlatiladigan model: {ACTIVE_MODEL}")

"""## 🤖 4. AI test yaratish funksiyasi"""

class QuizError(Exception):
    """Foydalanuvchiga ko'rsatiladigan tushunarli xatolik."""


SCHEMA_TEXT = (
    '[ {"question": "Savol matni", '
    '"options": {"A": "1-javob", "B": "2-javob", "C": "3-javob", "D": "4-javob"}, '
    '"correct_answer": "A", "explanation": "Savolning tushuntirishi"} ]'
)


def friendly_api_error(exc: Exception) -> str:
    """API xatolarini sodda tilda tushuntiradi."""
    text = str(exc)
    low = text.lower()
    if "api key not valid" in low or "api_key_invalid" in low or "permission_denied" in low or "401" in text or "403" in text:
        return "🔑 API kalit noto'g'ri yoki ruxsati yo'q. Secrets'dagi GEMINI_API_KEY ni tekshiring."
    if "429" in text or "resource_exhausted" in low or "quota" in low:
        return "⏳ So'rovlar limiti tugadi. Birozdan keyin qayta urinib ko'ring."
    if "404" in text or "not_found" in low or "not found" in low:
        return f"🧩 '{ACTIVE_MODEL}' modeli topilmadi. Konfiguratsiya cell'idagi MODEL ni o'zgartiring."
    if "503" in text or "unavailable" in low or "overloaded" in low or "500" in text:
        return "🌐 Gemini serveri hozir band. Bir ozdan keyin qayta urinib ko'ring."
    if "timeout" in low or "timed out" in low or "connection" in low:
        return "📡 Internet aloqasida muammo. Ulanishni tekshirib, qayta urinib ko'ring."
    return f"⚠️ Kutilmagan xatolik: {text[:200]}"


def build_prompt(topic: str, difficulty: str, count: int, existing: list) -> str:
    prompt = (
        "Sen tajribali IT o'qituvchisan va test tuzuvchisan.\n"
        f"Quyidagi shartlarga mos {count} ta ko'p tanlovli savol tuz.\n\n"
        f"Mavzu: {topic}\n"
        f"Qiyinlik darajasi: {difficulty} ({DIFFICULTY_HINTS[difficulty]})\n\n"
        "Talablar:\n"
        "- Savollar, variantlar va tushuntirishlar o'zbek tilida bo'lsin (texnik atamalar va kod inglizcha qolishi mumkin).\n"
        "- Har bir savolda aynan 4 ta variant (A, B, C, D) bo'lsin va faqat bittasi to'g'ri bo'lsin.\n"
        "- Noto'g'ri variantlar ishonarli, lekin aniq xato bo'lsin.\n"
        "- Savollar bir-birini takrorlamasin va mavzuning turli qismlarini qamrab olsin.\n"
        "- 'explanation' maydonida variant harflariga (A, B, C, D) havola qilma; javob nima uchun to'g'ri ekanini 1-3 gapda tushuntir.\n"
        "- Javob FAQAT yaroqli JSON massiv bo'lsin, boshqa matn yoki Markdown bo'lmasin.\n\n"
        "Har bir element formati:\n" + SCHEMA_TEXT + "\n"
    )
    if existing:
        prompt += "\nQuyidagi savollarni takrorlama:\n" + "\n".join(f"- {q}" for q in existing[:30]) + "\n"
    return prompt


def request_questions(topic: str, difficulty: str, count: int, existing: list) -> str:
    """Gemini'dan xom JSON matnini oladi."""
    try:
        response = client.models.generate_content(
            model=ACTIVE_MODEL,
            contents=build_prompt(topic, difficulty, count, existing),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.9,
            ),
        )
    except Exception as exc:
        raise QuizError(friendly_api_error(exc)) from exc
    text = getattr(response, "text", None)
    if not text:
        raise QuizError("🤖 AI bo'sh javob qaytardi. Qayta urinib ko'ring.")
    return text


def generate_questions(topic: str, difficulty: str, count: int) -> list:
    """Savollarni yaratadi, validatsiya qiladi, takrorlarni olib tashlaydi."""
    collected, seen = [], set()
    last_problem = ""
    for _ in range(MAX_ATTEMPTS):
        need = count - len(collected)
        if need <= 0:
            break
        raw = request_questions(topic, difficulty, need, [q["question"] for q in collected])
        try:
            items = parse_questions_json(raw)
        except ValueError as exc:
            last_problem = str(exc)
            continue
        for item in items:
            try:
                q = validate_question(item)
            except ValueError as exc:
                last_problem = str(exc)
                continue
            key = re.sub(r"\W+", " ", q["question"].lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            collected.append(shuffle_options(q))
    if len(collected) < count:
        raise QuizError(
            f"🤖 AI {count} ta to'g'ri formatdagi savol bera olmadi (tayyor: {len(collected)}). "
            f"Qayta urinib ko'ring. {last_problem}".strip()
        )
    return collected[:count]

"""## ✅ 5. Test validatsiyasi
AI javobi JSON sifatida o'qiladi, har bir savol qat'iy tekshiriladi, so'ng variantlar tasodifiy aralashtiriladi (to'g'ri javob doim A bo'lib qolmasligi uchun).
"""

def parse_questions_json(raw: str) -> list:
    """AI matnidan savollar ro'yxatini ajratib oladi."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            raise ValueError("AI javobi JSON formatida emas.")
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("AI javobidagi JSON buzilgan.") from exc
    if isinstance(data, dict):
        for key in ("questions", "quiz", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError("AI javobida savollar ro'yxati yo'q.")
    return data


def validate_question(item) -> dict:
    """Bitta savolni tekshiradi va tozalangan lug'at qaytaradi. Xato bo'lsa ValueError."""
    if not isinstance(item, dict):
        raise ValueError("Savol obyekt emas.")
    question = str(item.get("question", "")).strip()
    if not question:
        raise ValueError("Savol matni bo'sh.")

    raw_options = item.get("options")
    if isinstance(raw_options, list) and len(raw_options) == 4:
        raw_options = dict(zip(LETTERS, raw_options))
    if not isinstance(raw_options, dict):
        raise ValueError("Variantlar noto'g'ri formatda.")
    options = {str(k).strip().upper(): str(v).strip() for k, v in raw_options.items()}
    if sorted(options.keys()) != LETTERS:
        raise ValueError("Variantlar aynan A, B, C, D bo'lishi kerak.")
    if any(not v for v in options.values()):
        raise ValueError("Bo'sh variant bor.")
    if len({v.lower() for v in options.values()}) != 4:
        raise ValueError("Variantlar takrorlangan.")

    correct = str(item.get("correct_answer", "")).strip().upper()[:1]
    if correct not in LETTERS:
        raise ValueError("To'g'ri javob A, B, C yoki D bo'lishi kerak.")

    explanation = str(item.get("explanation", "")).strip()
    if not explanation:
        raise ValueError("Tushuntirish bo'sh.")

    return {"question": question, "options": options, "correct_answer": correct, "explanation": explanation}


def shuffle_options(q: dict) -> dict:
    """Variantlarni aralashtiradi va correct_answer harfini mos ravishda yangilaydi."""
    correct_text = q["options"][q["correct_answer"]]
    texts = [q["options"][k] for k in LETTERS]
    random.shuffle(texts)
    options = dict(zip(LETTERS, texts))
    new_correct = next(k for k, v in options.items() if v == correct_text)
    return {
        "question": q["question"],
        "options": options,
        "correct_answer": new_correct,
        "explanation": q["explanation"],
    }

"""## 📝 6. Practice Mode logikasi
Javob tekshirilmaguncha keyingi savolga o'tib bo'lmaydi.
"""

def new_session(questions: list, topic: str, difficulty: str, mode: str) -> dict:
    """Yangi test sessiyasi holatini yaratadi."""
    return {
        "questions": questions,
        "topic": topic,
        "difficulty": difficulty,
        "mode": mode,
        "index": 0,
        "answers": {},     # {savol_indeksi: "A"}
        "checked": set(),  # Practice Mode'da tekshirilgan savollar
        "screen": "quiz",
    }


def reset_session(s: dict) -> dict:
    """Xuddi shu testni boshidan boshlaydi."""
    s["index"] = 0
    s["answers"] = {}
    s["checked"] = set()
    s["screen"] = "quiz"
    return s


def practice_check(s: dict, selected):
    """Javobni tekshiradi. (muvaffaqiyat, xabar) qaytaradi."""
    if selected not in LETTERS:
        return False, "Avval javob variantini tanlang."
    i = s["index"]
    if i in s["checked"]:
        return True, ""
    s["answers"][i] = selected
    s["checked"].add(i)
    return True, ""


def practice_next(s: dict):
    """Keyingi savolga o'tadi; oxirgi savoldan keyin natijalarga olib boradi."""
    i = s["index"]
    if i not in s["checked"]:
        return False, "Keyingi savolga o'tish uchun avval 'Javobni tekshirish' tugmasini bosing."
    if i < len(s["questions"]) - 1:
        s["index"] += 1
    else:
        s["screen"] = "result"
    return True, ""

"""## 🎓 7. Exam Mode logikasi
Javoblar saqlanadi, natija test oxirigacha ko'rsatilmaydi. Yakunlashdan oldin barcha savollar javoblanganligi tekshiriladi.
"""

def exam_save(s: dict, selected) -> bool:
    """Tanlangan javobni saqlaydi."""
    if selected in LETTERS:
        s["answers"][s["index"]] = selected
        return True
    return False


def exam_move(s: dict, delta: int, selected) -> None:
    """Joriy javobni saqlab, savollar bo'ylab harakatlanadi."""
    exam_save(s, selected)
    s["index"] = max(0, min(len(s["questions"]) - 1, s["index"] + delta))


def exam_unanswered(s: dict) -> list:
    """Javobsiz savollarning raqamlari (1 dan boshlab)."""
    return [i + 1 for i in range(len(s["questions"])) if i not in s["answers"]]


def exam_finish(s: dict, selected):
    """Testni yakunlaydi. Javobsiz savollar bo'lsa, ularning ro'yxatini qaytaradi."""
    exam_save(s, selected)
    missing = exam_unanswered(s)
    if missing:
        return False, missing
    s["screen"] = "result"
    return True, []

"""## 📊 8. Natijalarni hisoblash"""

def get_grade(percent: float):
    for threshold, label, color in GRADES:
        if percent >= threshold:
            return label, color
    return GRADES[-1][1], GRADES[-1][2]


def compute_results(s: dict) -> dict:
    """Jami natijalar va har bir savol bo'yicha tahlilni hisoblaydi."""
    details, correct = [], 0
    for i, q in enumerate(s["questions"]):
        chosen = s["answers"].get(i)
        is_ok = chosen == q["correct_answer"]
        correct += int(is_ok)
        details.append({
            "number": i + 1,
            "question": q["question"],
            "chosen": chosen,
            "chosen_text": q["options"].get(chosen) if chosen else None,
            "correct": q["correct_answer"],
            "correct_text": q["options"][q["correct_answer"]],
            "is_correct": is_ok,
            "explanation": q["explanation"],
        })
    total = len(s["questions"])
    percent = round(correct / total * 100, 1) if total else 0.0
    label, color = get_grade(percent)
    return {
        "total": total, "correct": correct, "wrong": total - correct,
        "percent": percent, "grade": label, "color": color,
        "topic": s["topic"], "difficulty": s["difficulty"], "mode": s["mode"],
        "details": details,
    }

"""## 🎨 9. Gradio UI
Zamonaviy dark SaaS dashboard dizayni (CSS orqali).
"""

import gradio as gr
import html
import re
import traceback

CSS = """
:root, .dark, .gradio-container {
  --body-background-fill: #070b16;
  --background-fill-primary: #0d1424;
  --background-fill-secondary: #0b1120;
  --block-background-fill: transparent;
  --block-border-color: transparent;
  --block-border-width: 0px;
  --panel-background-fill: #0d1424;
  --body-text-color: #e2e8f0;
  --body-text-color-subdued: #94a3b8;
  --block-label-text-color: #94a3b8;
  --block-label-background-fill: transparent;
  --block-title-text-color: #cbd5e1;
  --border-color-primary: #1e293b;
  --input-background-fill: #0b1120;
  --input-border-color: #1f2a44;
  --input-border-color-focus: #22d3ee;
  --input-shadow-focus: 0 0 0 3px rgba(34,211,238,.18);
  --checkbox-label-background-fill: #0b1120;
  --checkbox-label-background-fill-hover: #121b31;
  --checkbox-label-background-fill-selected: rgba(34,211,238,.12);
  --checkbox-label-border-color: #1f2a44;
  --checkbox-label-border-color-hover: #22d3ee;
  --checkbox-label-border-color-selected: #22d3ee;
  --checkbox-label-text-color: #e2e8f0;
  --checkbox-label-text-color-selected: #ecfeff;
  --button-secondary-background-fill: #111a2e;
  --button-secondary-background-fill-hover: #17223b;
  --button-secondary-text-color: #e2e8f0;
  --button-secondary-border-color: #24324f;
  --radius-lg: 14px; --radius-md: 12px;
  color-scheme: dark;
}
body, .gradio-container {
  background: radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,.12), transparent 60%),
              radial-gradient(900px 500px at 100% 0%, rgba(139,92,246,.14), transparent 55%), #070b16 !important;
}
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
footer { display: none !important; }

.cl-hero { display:flex; align-items:center; gap:18px; flex-wrap:wrap; padding:22px 26px; border-radius:20px;
  background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(99,102,241,.12) 55%, rgba(139,92,246,.12));
  border:1px solid rgba(99,102,241,.35); box-shadow: 0 10px 40px rgba(2,6,23,.55); }
.cl-logo { width:58px; height:58px; border-radius:16px; display:flex; align-items:center; justify-content:center;
  font: 700 22px ui-monospace, monospace; color:#fff; background:linear-gradient(135deg,#06b6d4,#6366f1 60%,#8b5cf6);
  box-shadow:0 8px 24px rgba(99,102,241,.45); }
.cl-title { font-size:30px; font-weight:800; letter-spacing:-.5px;
  background:linear-gradient(90deg,#67e8f9,#a5b4fc,#c4b5fd); -webkit-background-clip:text; background-clip:text; color:transparent; }
.cl-tag { color:#94a3b8; font-size:14px; margin-top:2px; }
.cl-hero-desc { margin-left:auto; max-width:380px; color:#9fb0c8; font-size:13.5px; line-height:1.55; }

.cl-card { background:#0d1424; border:1px solid #1b2742; border-radius:18px; padding:18px 20px;
  box-shadow:0 6px 24px rgba(2,6,23,.45); }
.cl-h { font-size:15px; font-weight:700; color:#e2e8f0; margin:0 0 6px 0; display:flex; align-items:center; gap:8px; }
.cl-sub { color:#8da0bb; font-size:13px; line-height:1.55; }
.cl-modes { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:6px; }
.cl-mode { border:1px solid #1f2a44; border-radius:12px; padding:10px 12px; background:#0b1120; font-size:12.5px; color:#9fb0c8; }
.cl-mode b { color:#e2e8f0; display:block; margin-bottom:2px; }

.cl-chips { display:flex; gap:8px; flex-wrap:wrap; }
.cl-chip { font-size:12px; padding:4px 10px; border-radius:999px; background:#111a2e; border:1px solid #24324f; color:#b6c4da; }
.cl-chip.accent { background:rgba(34,211,238,.10); border-color:rgba(34,211,238,.4); color:#67e8f9; }
.cl-prog-top { display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
.cl-qnum { font-weight:700; color:#e2e8f0; }
.cl-bar { height:8px; border-radius:999px; background:#111a2e; overflow:hidden; margin-top:12px; }
.cl-bar > div { height:100%; border-radius:999px; background:linear-gradient(90deg,#06b6d4,#6366f1,#8b5cf6); transition:width .4s; }

.cl-question { font-size:19px; line-height:1.55; font-weight:600; color:#f1f5f9; }
.cl-question code, .cl-ans code { background:#111a2e; border:1px solid #24324f; padding:1px 6px; border-radius:6px;
  font-family: ui-monospace, monospace; font-size:.9em; color:#7dd3fc; }
.cl-qbadge { display:inline-block; font-size:12px; font-weight:700; padding:3px 10px; border-radius:8px;
  background:linear-gradient(135deg,#06b6d4,#6366f1); color:#fff; margin-bottom:10px; }

.cl-opts label { border-radius:12px !important; padding:12px 14px !important; margin-bottom:8px; width:100%; cursor:pointer; transition:all .15s; }
.cl-opts label:hover { transform: translateX(3px); }
.cl-opts label span { font-size:15px; }
.cl-opts .wrap { gap:0 !important; display:block !important; }

.cl-note { border-radius:12px; padding:12px 14px; font-size:14px; line-height:1.55; margin:4px 0; }
.cl-note.ok   { background:rgba(34,197,94,.10);  border:1px solid rgba(34,197,94,.45);  color:#bbf7d0; }
.cl-note.bad  { background:rgba(239,68,68,.10);  border:1px solid rgba(239,68,68,.45);  color:#fecaca; }
.cl-note.warn { background:rgba(245,158,11,.10); border:1px solid rgba(245,158,11,.45); color:#fde68a; }
.cl-note.err  { background:rgba(239,68,68,.10);  border:1px solid rgba(239,68,68,.45);  color:#fecaca; }
.cl-note.info { background:rgba(34,211,238,.08); border:1px solid rgba(34,211,238,.35); color:#a5f3fc; }
.cl-note .cl-expl { margin-top:8px; color:#e2e8f0; }

button.primary { background:linear-gradient(135deg,#06b6d4,#6366f1 70%,#8b5cf6) !important; border:none !important;
  color:#fff !important; font-weight:700 !important; box-shadow:0 8px 22px rgba(99,102,241,.35); }
button.primary:hover { filter:brightness(1.1); }
button.secondary { border-radius:12px !important; }

.cl-result { display:flex; gap:26px; align-items:center; flex-wrap:wrap; }
.cl-ring { width:150px; height:150px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.cl-ring-in { width:116px; height:116px; border-radius:50%; background:#0d1424; display:flex; flex-direction:column;
  align-items:center; justify-content:center; }
.cl-ring-in b { font-size:30px; color:#f8fafc; } .cl-ring-in span { font-size:12px; color:#8da0bb; }
.cl-grade h2 { margin:2px 0 6px 0; font-size:30px; font-weight:800; }
.cl-grade .cl-sub { font-size:13px; }
.cl-stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-top:16px; }
.cl-stat { background:#0b1120; border:1px solid #1f2a44; border-radius:14px; padding:12px 14px; }
.cl-stat span { font-size:12px; color:#8da0bb; display:block; } .cl-stat b { font-size:24px; color:#f1f5f9; }

.cl-item { border-radius:14px; padding:14px 16px; margin-bottom:12px; background:#0b1120; border:1px solid #1f2a44; border-left-width:5px; }
.cl-item.ok { border-left-color:#22c55e; } .cl-item.bad { border-left-color:#ef4444; }
.cl-item-top { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px; }
.cl-badge { font-size:12px; font-weight:700; padding:3px 10px; border-radius:999px; }
.cl-badge.ok { background:rgba(34,197,94,.15); color:#86efac; } .cl-badge.bad { background:rgba(239,68,68,.15); color:#fca5a5; }
.cl-ans { font-size:14px; color:#cbd5e1; margin:3px 0; line-height:1.5; }
.cl-ans b { color:#94a3b8; font-weight:600; }
.cl-expl2 { margin-top:8px; padding:10px 12px; border-radius:10px; background:rgba(99,102,241,.08);
  border:1px solid rgba(99,102,241,.25); color:#c7d2fe; font-size:13.5px; line-height:1.55; }

@media (max-width: 720px) {
  .cl-hero-desc { margin-left:0; } .cl-modes { grid-template-columns:1fr; } .cl-question { font-size:17px; }
}
"""

THEME = gr.themes.Base(
    primary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


# ---------- HTML yordamchilari ----------
def fmt(text) -> str:
    """Matnni xavfsiz HTML'ga aylantiradi (`kod` va yangi qatorlarni qo'llab-quvvatlaydi)."""
    t = html.escape(str(text))
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t.replace("\n", "<br>")


def note(kind: str, message: str) -> str:
    return f'<div class="cl-note {kind}">{message}</div>'


HERO_HTML = """
<div class="cl-hero">
  <div class="cl-logo">&lt;/&gt;</div>
  <div>
    <div class="cl-title">AI CodeLab</div>
    <div class="cl-tag">Test your knowledge. Discover your potential.</div>
  </div>
  <div class="cl-hero-desc">🤖 Sun\'iy intellekt yordamida IT mavzularida o\\'z testingizni yarating,
  bilimingizni sinang va natijalaringizni batafsil tahlil qiling.</div>
</div>
"""

MODE_INFO_HTML = """
<div class="cl-modes">
  <div class="cl-mode"><b>🎯 Practice Mode</b>Har bir javobdan keyin darhol natija va AI tushuntirishi ko\\'rsatiladi.</div>
  <div class="cl-mode"><b>🎓 Exam Mode</b>Natijalar faqat test oxirida chiqadi. Savollar orasida erkin yurish mumkin.</div>
</div>
"""


def render_progress(s: dict) -> str:
    n, i = len(s["questions"]), s["index"]
    practice = s["mode"] == PRACTICE
    done = len(s["checked"]) if practice else len(s["answers"])
    pct = int(done / n * 100) if n else 0
    label = "Tekshirilgan" if practice else "Javob berilgan"
    return f"""
<div class="cl-card">
  <div class="cl-prog-top">
    <div class="cl-qnum">📍 Savol {i + 1} / {n}</div>
    <div class="cl-chips">
      <span class="cl-chip accent">{html.escape(s['topic'])}</span>
      <span class="cl-chip">{html.escape(s['difficulty'])}</span>
      <span class="cl-chip">{html.escape(s['mode'])}</span>
      <span class="cl-chip">{label}: {done}/{n}</span>
    </div>
  </div>
  <div class="cl-bar"><div style="width:{pct}%"></div></div>
</div>"""


def render_question(s: dict) -> str:
    q = s["questions"][s["index"]]
    return f"""
<div class="cl-card">
  <span class="cl-qbadge">Savol #{s['index'] + 1}</span>
  <div class="cl-question">{fmt(q['question'])}</div>
</div>"""


def render_feedback(s: dict, notice: str = "") -> str:
    i = s["index"]
    q = s["questions"][i]
    parts = [notice] if notice else []
    if s["mode"] == PRACTICE and i in s["checked"]:
        chosen, correct = s["answers"].get(i), q["correct_answer"]
        if chosen == correct:
            head, kind = "✅ To\\'g\'ri javob!", "ok"
        else:
            head, kind = f"❌ Noto\\'g\'ri. Siz <b>{chosen}</b> ni tanladingiz.", "bad"
        parts.append(note(kind, (
            f"<b>{head}</b><br>To\\'g\'ri javob: <b>{correct}. {fmt(q['options'][correct])}</b>"
            f"<div class=\\'cl-expl\\'>💡 <b>Tushuntirish:</b> {fmt(q['explanation'])}</div>"
        )))
    elif s["mode"] == EXAM and i in s["answers"]:
        parts.append(note("info", f"💾 Javobingiz saqlandi: <b>{s['answers'][i]}</b>"))
    return "".join(parts)


def render_summary(r: dict) -> str:
    deg = r["percent"] * 3.6
    return f"""
<div class="cl-card">
  <div class="cl-result">
    <div class="cl-ring" style="background:conic-gradient({r['color']} {deg}deg, rgba(148,163,184,.18) 0)">
      <div class="cl-ring-in"><b>{r['percent']:g}%</b><span>natija</span></div>
    </div>
    <div class="cl-grade">
      <div class="cl-sub">Yakuniy baholash</div>
      <h2 style="color:{r['color']}">{html.escape(r['grade'])}</h2>
      <div class="cl-chips">
        <span class="cl-chip accent">📚 {html.escape(r['topic'])}</span>
        <span class="cl-chip">🎯 {html.escape(r['difficulty'])}</span>
        <span class="cl-chip">🧭 {html.escape(r['mode'])}</span>
      </div>
    </div>
  </div>
  <div class="cl-stats">
    <div class="cl-stat"><span>Jami savollar</span><b>{r['total']}</b></div>
    <div class="cl-stat"><span>✅ To\\'g\'ri javoblar</span><b style="color:#4ade80">{r['correct']}</b></div>
    <div class="cl-stat"><span>❌ Noto\\'g\'ri javoblar</span><b style="color:#f87171">{r['wrong']}</b></div>
    <div class="cl-stat"><span>📈 Natija foizi</span><b>{r['percent']:g}%</b></div>
  </div>
</div>"""


def render_analysis(r: dict) -> str:
    cards = ['<div class="cl-card"><div class="cl-h">🔍 Javoblar tahlili</div><div style="margin-top:12px">']
    for d in r["details"]:
        kind = "ok" if d["is_correct"] else "bad"
        status = "✅ To'g'ri" if d["is_correct"] else "❌ Noto'g'ri"
        if d["chosen"]:
            chosen = f"{d['chosen']}. {fmt(d['chosen_text'])}"
        else:
            chosen = "— javob berilmagan"
        cards.append(f"""
<div class="cl-item {kind}">
  <div class="cl-item-top"><b>Savol {d['number']}</b><span class="cl-badge {kind}">{status}</span></div>
  <div class="cl-ans" style="color:#f1f5f9;font-weight:600">{fmt(d['question'])}</div>
  <div class="cl-ans"><b>Sizning javobingiz:</b> {chosen}</div>
  <div class="cl-ans"><b>To'g'ri javob:</b> {d['correct']}. {fmt(d['correct_text'])}</div>
  <div class="cl-expl2">💡 {fmt(d['explanation'])}</div>
</div>""")
    cards.append("</div></div>")
    return "".join(cards)


# ---------- Gradio komponentlari ----------
with gr.Blocks(title="AI CodeLab", css=CSS, theme=THEME) as demo:
    state = gr.State(None)
    gr.HTML(HERO_HTML)

    # --- Sozlamalar ekrani ---
    with gr.Column(visible=True) as setup_col:
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320, elem_classes="cl-card"):
                gr.HTML('<div class="cl-h">🧪 Test yaratish</div><div class="cl-sub">Mavzu va qiyinlik darajasini tanlang.</div>')
                topic = gr.Dropdown(TOPICS + [OTHER_TOPIC], value=TOPICS[0], label="📚 Test mavzusi")
                custom_topic = gr.Textbox(label="✍️ O\'z mavzuingiz", placeholder="Masalan: Docker, Git, Linux...", visible=False)
                difficulty = gr.Radio(DIFFICULTIES, value=DIFFICULTIES[1], label="🎯 Qiyinlik darajasi")
            with gr.Column(scale=1, min_width=320, elem_classes="cl-card"):
                gr.HTML('<div class="cl-h">⚙️ Test sozlamalari</div><div class="cl-sub">Boshlashdan oldin istalgan sozlamani o\'quvchi o\'zgartirishi mumkin.</div>')
                count = gr.Radio(QUESTION_COUNTS, value=10, label="🔢 Savollar soni")
                mode = gr.Radio(MODES, value=PRACTICE, label="🧭 Test rejimi")
                gr.HTML(MODE_INFO_HTML)
        setup_msg = gr.HTML("")
        start_btn = gr.Button("🚀 Testni boshlash", variant="primary", size="lg")

    # --- Test ekrani ---
    with gr.Column(visible=False) as quiz_col:
        progress_html = gr.HTML()
        question_html = gr.HTML()
        options = gr.Radio(choices=[], value=None, label="Javobingizni tanlang", elem_classes="cl-opts")
        feedback_html = gr.HTML()
        with gr.Row():
            check_btn = gr.Button("✅ Javobni tekshirish", variant="primary")
            save_btn = gr.Button("💾 Javobni saqlash", variant="secondary")
            prev_btn = gr.Button("⬅️ Oldingi", variant="secondary")
            next_btn = gr.Button("Keyingi savol ➡️", variant="primary")
            finish_btn = gr.Button("🏁 Testni yakunlash", variant="primary")
        with gr.Row():
            quiz_restart_btn = gr.Button("🔄 Testni qayta boshlash", variant="secondary", size="sm")
            quiz_home_btn = gr.Button("🏠 Yangi test yaratish", variant="secondary", size="sm")

    # --- Natijalar ekrani ---
    with gr.Column(visible=False) as result_col:
        summary_html = gr.HTML()
        analysis_html = gr.HTML()
        with gr.Row():
            res_restart_btn = gr.Button("🔄 Testni qayta boshlash", variant="secondary")
            res_new_btn = gr.Button("✨ Yangi test yaratish", variant="primary")

    # Barcha callback'lar shu ro'yxatga yozadi
    OUTPUTS = [
        state, setup_col, quiz_col, result_col, setup_msg,
        progress_html, question_html, options, feedback_html,
        check_btn, save_btn, prev_btn, next_btn, finish_btn,
        summary_html, analysis_html,
    ]
    N_QUIZ_PART = 9
    N_RESULT_PART = 2

    # ---------- Ko\'rinishni boshqarish ----------
    def quiz_updates(s: dict, notice: str = "") -> list:
        i, n = s["index"], len(s["questions"])
        q = s["questions"][i]
        practice = s["mode"] == PRACTICE
        checked = i in s["checked"]
        last = i == n - 1
        return [
            render_progress(s),
            render_question(s),
            gr.update(
                choices=[(f"{k}. {q['options'][k]}", k) for k in LETTERS],
                value=s["answers"].get(i),
                interactive=not (practice and checked),
            ),
            render_feedback(s, notice),
            gr.update(visible=practice, interactive=not checked),
            gr.update(visible=not practice),
            gr.update(visible=not practice, interactive=i > 0),
            gr.update(visible=True if practice else not last,
                      value=("🏁 Natijalarni ko'rish" if last else "Keyingi savol ➡️") if practice else "Keyingi ➡️"),
            gr.update(visible=not practice),
        ]


    def view(s=None, setup_message: str = "", notice: str = "") -> list:
        screen = s["screen"] if s else "setup"
        out = [
            s,
            gr.update(visible=screen == "setup"),
            gr.update(visible=screen == "quiz"),
            gr.update(visible=screen == "result"),
            note("err", html.escape(setup_message).replace("\n", "<br>")) if setup_message else "",
        ]
        out += quiz_updates(s, notice) if screen == "quiz" else [gr.update()] * N_QUIZ_PART
        if screen == "result":
            r = compute_results(s)
            out += [render_summary(r), render_analysis(r)]
        else:
            out += [gr.update()] * N_RESULT_PART
        assert len(out) == len(OUTPUTS), "view() chiqishlari soni OUTPUTS ga mos emas"
        return out


    # ---------- Callback\'lar ----------
    def on_topic_change(value):
        return gr.update(visible=(value == OTHER_TOPIC))


    def on_start(topic_v, custom_v, difficulty_v, count_v, mode_v, s):
        topic_name = (custom_v or "").strip()[:100] if topic_v == OTHER_TOPIC else topic_v
        if not topic_name:
            return view(None, setup_message="✍️ Iltimos, o'z mavzuingizni yozing.")
        try:
            questions = generate_questions(topic_name, difficulty_v, int(count_v))
        except QuizError as exc:
            return view(None, setup_message=str(exc))
        except Exception as exc:
            traceback.print_exc()
            return view(None, setup_message=friendly_api_error(exc))
        return view(new_session(questions, topic_name, difficulty_v, mode_v))


    def on_check(s, selected):
        if not s or s["screen"] != "quiz" or s["mode"] != PRACTICE:
            return view(s)
        ok, msg = practice_check(s, selected)
        return view(s, notice="" if ok else note("warn", msg))


    def on_save(s, selected):
        if not s or s["screen"] != "quiz" or s["mode"] != EXAM:
            return view(s)
        if not exam_save(s, selected):
            return view(s, notice=note("warn", "Avval javob variantini tanlang."))
        return view(s)


    def on_prev(s, selected):
        if not s or s["screen"] != "quiz" or s["mode"] != EXAM:
            return view(s)
        exam_move(s, -1, selected)
        return view(s)


    def on_next(s, selected):
        if not s or s["screen"] != "quiz":
            return view(s)
        if s["mode"] == PRACTICE:
            ok, msg = practice_next(s)
            return view(s, notice="" if ok else note("warn", msg))
        exam_move(s, +1, selected)
        return view(s)


    def on_finish(s, selected):
        if not s or s["screen"] != "quiz" or s["mode"] != EXAM:
            return view(s)
        ok, missing = exam_finish(s, selected)
        if not ok:
            nums = ", ".join(str(m) for m in missing)
            gr.Warning(f"Javobsiz savollar bor: {nums}")
            return view(s, notice=note("warn", f"⚠️ Testni yakunlash uchun barcha savollarga javob bering. Javobsiz savollar: <b>{nums}</b>"))
        return view(s)


    def on_restart(s):
        if not s:
            return view(None)
        return view(reset_session(s))


    def on_new(s):
        return view(None)


    # ---------- Event Bindings ----------
    topic.change(on_topic_change, inputs=[topic], outputs=[custom_topic])
    start_btn.click(on_start, inputs=[topic, custom_topic, difficulty, count, mode, state], outputs=OUTPUTS)
    check_btn.click(on_check, inputs=[state, options], outputs=OUTPUTS)
    save_btn.click(on_save, inputs=[state, options], outputs=OUTPUTS)
    prev_btn.click(on_prev, inputs=[state, options], outputs=OUTPUTS)
    next_btn.click(on_next, inputs=[state, options], outputs=OUTPUTS)
    finish_btn.click(on_finish, inputs=[state, options], outputs=OUTPUTS)
    quiz_restart_btn.click(on_restart, inputs=[state], outputs=OUTPUTS)
    res_restart_btn.click(on_restart, inputs=[state], outputs=OUTPUTS)
    quiz_home_btn.click(on_new, inputs=[state], outputs=OUTPUTS)
    res_new_btn.click(on_new, inputs=[state], outputs=OUTPUTS)

print("✅ UI tayyor.")

"""## ▶️ Dasturni ishga tushirish"""

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "7860")),
    share=False,
)