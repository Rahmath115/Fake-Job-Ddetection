import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("🛡 GENCRAFT SafeHire")
st.subheader("AI-Powered Fake Job Detection System")
st.caption("Developed by THE GENCRAFT")
st.write("Enter job description below 👇")

# Input box
user_input = st.text_area("Job Description")

# Button
if st.button("Check Job"):

    if user_input.strip() == "":
        st.warning("Please enter job description")

    else:
        # STEP 1: Convert text → numbers
        transformed = vectorizer.transform([user_input])

        # STEP 2: Predict
        prediction = model.predict(transformed)

        # STEP 3: Show result
        if prediction[0] == 1:
            st.error("🚨 This is a FAKE Job")

        else:
            st.success("🟩 This is a REAL Job")

            st.markdown("### 🔎 Explore Similar Genuine Jobs On Trusted Platforms:")

            st.markdown("""
- 🟢 LinkedIn Jobs  
- 🟢 Indeed  
- 🟢 Naukri  
- 🟢 Glassdoor  
- 🟢 Official Company Career Pages  
""")

            st.info("Always apply through official company websites or verified job portals to stay safe.")