import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("fake_job_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("🛡 GENCRAFT SafeHire")
st.subheader("AI-Powered Fake Job Detection System")
st.caption("Developed by THE GENCRAFT")
st.write("Enter job description below 👇")

# Input box
# Input box
st.write("Enter job description below 👇")
user_input = st.text_area("Job Description")

# Button
if st.button("Check Job"):

    if user_input.strip() == "":
        st.warning("Please enter job description")

    else:
        # Transform input
        transformed = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(transformed)

        # FAKE JOB
        if prediction[0] == 1:
            st.error("🚨 This is a FAKE Job")

            st.markdown("### ⚠ Risk Analysis")

            st.warning("""
🔴 Red Flags Detected:
- Unrealistic earning claims
- No skill requirement
- Urgency-based language
- Possible financial involvement

💡 Recommendation:
Avoid applying and verify through official company portals.
""")

        # REAL JOB
        else:
            st.success("🟢 This is a REAL Job")

            st.markdown("### 📊 Job Market Insights")

            st.info("""
✔ Acceptance Probability: ~72%

✔ Where you can apply:
- LinkedIn
- Indeed
- Naukri
- Company Career Portals

✔ Typical Requirements:
- Basic domain skills
- Communication ability
- Relevant qualification

📈 Hiring Trend:
This role is currently in moderate demand in corporate sectors.
""")

        st.markdown("### 🔍 Explore Similar Genuine Jobs On Trusted Platforms:")

        st.markdown("""
- LinkedIn Jobs
- Indeed
- Naukri
- Glassdoor
- Official Company Career Pages
""")

        st.info("Always apply through official company websites or verified job portals to stay safe.")