import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
import PyPDF2


# ==============================
# CONFIG
# ==============================
# Ensure the upload folder exists for temporary processing
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Replace with your actual Gemini API key
# BEST PRACTICE: Set this in your environment variables as 'GEMINI_API_KEY'
# Get your key here: https://aistudio.google.com/app/apikey
apiKey = os.getenv("GEMINI_API_KEY")

# For local development, if you don't have env vars set, you can paste it here:
if not apiKey:
    # Fallback to the provided key
    apiKey = ""

try:
    client = genai.Client(api_key=apiKey)
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Gemini Client. Check your API Key. Details: {e}")
    client = None

app = Flask(__name__)
# Enable CORS so the browser (frontend) can talk to this server without security blocks
CORS(app) 
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# PDF PARSING
# ==============================
def extract_text_from_pdf(pdf_path):
    """Parses PDF content into raw text."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"PDF Error: {e}")
    return text


# ==============================
# AI LOGIC
# ==============================
def parse_resume(resume_text):
    """Extracts key info from the resume using AI."""
    prompt = f"Extract skills, Experience, and Education from this resume in bullet points:\n{resume_text}"
    if not client:
         return "Error: AI Client not initialized."
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def parse_job_description(jd_text):
    """Extracts requirements from the JD using AI."""
    prompt = f"Extract Required Skills and Qualifications from this Job Description:\n{jd_text}"
    if not client:
        return "Error: AI Client not initialized."
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def get_final_json_analysis(parsed_resume, parsed_jd):
    """Aggregates data into the specific JSON format the Scoreboard UI requires."""
    prompt = f"""
    Based on the parsed Resume and Job Description below, generate a final ATS evaluation.
    
    Resume Info: {parsed_resume}
    JD Info: {parsed_jd}

    Return ONLY a JSON object with this structure:
    {{
      "score": (integer 0-100),
      "summary": "Short 2-sentence match summary",
      "feedback": {{
        "verdict": "A professional paragraph assessing the candidate's fit",
        "pursueSkills": ["Skill 1", "Skill 2", "Skill 3"],
        "nextSteps": "Specific advice"
      }}
    }}
    """
    if not client:
        # Return a mock JSON if AI is broken so the app doesn't crash
        return json.dumps({
            "score": 0,
            "summary": "AI Error: Client not connected.",
            "feedback": {
                "verdict": "Could not analyze due to server configuration error.",
                "pursueSkills": [],
                "nextSteps": "Check server logs."
            }
        })

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return response.text

# ==============================
# ROUTES (Connecting Frontend & Backend)
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    The Bridge: Receives the FormData from the frontend 'fetch' call.
    Expects: 'resume' (PDF file) and 'job_description' (string).
    """
    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")
    
    print(f"DEBUG: Received analysis request. File: {resume_file.filename}, JD Length: {len(jd_text) if jd_text else 0}")

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    try:
        # 1. Process the file
        filename = f"temp_{resume_file.filename}"
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        resume_file.save(pdf_path)
        resume_text = extract_text_from_pdf(pdf_path)

        # 2. Run AI Analysis
        parsed_resume = parse_resume(resume_text)
        parsed_jd = parse_job_description(jd_text)

        # 3. Format result for the Scoreboard UI
        final_analysis_raw = get_final_json_analysis(parsed_resume, parsed_jd)
        final_data = json.loads(final_analysis_raw)

        # 4. Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return jsonify(final_data)

    except Exception as e:
        print(f"Internal Error: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    # Ensure port 8080 is used as per typical development environments
    app.run(debug=True, port=8080, host='0.0.0.0')
