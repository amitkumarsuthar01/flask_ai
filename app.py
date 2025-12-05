from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# ✅ Load STUDENT-FRIENDLY GPT-LIKE MODEL
print("✅ Loading Student Exam AI model (flan-t5-small)... Please wait...")
generator = pipeline("text2text-generation", model="google/flan-t5-large")
print("✅ AI Model Loaded Successfully!")

# Helper: remove repeated sentences
def remove_repeats(text):
    sentences = text.split(". ")
    seen = set()
    filtered = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            filtered.append(s)
            seen.add(s)
    return ". ".join(filtered)

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "Prompt required"}), 400

        # ✅ Optimized prompt for rich, student-friendly answers
        smart_prompt = f"""
Explain '{prompt}' in detail for a student.
- Give a clear definition.
- Describe the process step by step.
- Provide a real-world example.
- Make it engaging and easy to understand.
- Avoid repeating sentences.
"""

        # ✅ Generate answer
        result = generator(
            smart_prompt,
            max_new_tokens=600,  # allow longer answers
            do_sample=True,      # enable variation
            temperature=0.85,    # creativity
            top_k=50,
            top_p=0.95
        )

        # ✅ Remove repeated sentences
        final_answer = remove_repeats(result[0]["generated_text"])

        return jsonify({"reply": final_answer})

    except Exception as e:
        print("❌ AI Error:", str(e))
        return jsonify({"error": "AI processing failed"}), 500

# ✅ Health Check
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "✅ Student Exam AI Server is running!",
        "model": "google/flan-t5-large"
    })

if __name__ == "__main__":
    print("✅ Starting Python Student Exam AI Server on port 8000...")
    app.run(port=8000, debug=True)
